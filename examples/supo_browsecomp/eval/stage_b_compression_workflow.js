export const meta = {
  name: 'bcplus-compression-review',
  description: 'Diagnose whether BC+ eval failures are caused by context-compression info-loss or top_k overwhelm, then synthesize',
  phases: [
    { title: 'Diagnose', detail: 'one agent per compression-failure chain' },
    { title: 'Synthesize', detail: 'compression-loss taxonomy + per-checkpoint rates' },
  ],
}

// args = { points: [ { name, files:[...] } ] }
// Each chain JSON (sandbox-readable under /home) has: question, gold_answer,
// final_answer, bucket, n_sub_trajs, outcomes, summary_sources, objective flags
// (flag_dropped_by_compression / flag_summary_dropped_gold / flag_had_gold_early /
// flag_gold_in_final_ctx), gold_present_by_subtraj, gold_in_summary_by_subtraj, and
// sub_trajs[] each with: outcome, summary_source, topk_used, gold_windows (raw text
// around each gold occurrence in THAT sub-traj), summary (the FULL handover it wrote),
// response (truncated). The objective flags were computed by strict word-boundary
// substring match and MUST be confirmed/refuted by the agent.

const VERDICT = {
  type: 'object',
  additionalProperties: false,
  properties: {
    rollout_id: { type: ['integer', 'string', 'null'] },
    point: { type: 'string' },
    bucket: { type: 'string' },
    gold_really_present_early: {
      type: 'string',
      enum: ['yes-in-search-result', 'yes-in-opened-page', 'yes-in-reasoning', 'no-false-positive', 'unclear'],
      description: 'Verify the objective flag: was the gold answer ACTUALLY retrieved/known in an early sub-traj, or is the substring match a coincidence (e.g. a common word)?',
    },
    compression_loss: {
      type: 'string',
      enum: ['confirmed-answer-dropped', 'confirmed-lead-dropped', 'summary-distorted', 'no-loss-recovered', 'no-not-compression', 'unclear'],
      description: 'confirmed-answer-dropped = gold answer was in an early sub-traj and the <summary> handover failed to carry it forward, so it was lost; confirmed-lead-dropped = a strong lead/candidate (not literal gold) was dropped by a summary, derailing later work; summary-distorted = summary carried a fact but wrong/misleading; no-loss-recovered = summary dropped it but a later sub-traj re-found or still had it; no-not-compression = failure cause unrelated to compression.',
    },
    summary_quality: {
      type: 'string',
      enum: ['faithful', 'lossy-omitted-key-fact', 'distorted-wrong-fact', 'vague-unactionable', 'anchored-wrong-candidate', 'fallback-garbage'],
      description: 'Quality of the handover summary(ies) most relevant to the failure.',
    },
    topk_overwhelm: {
      type: 'string',
      enum: ['no', 'mild-large-obs', 'severe-drowned-signal'],
      description: 'Did large search results (high topk / many docs) bury the answer or accelerate compression? severe = the answer was likely present in a bloated observation the model skimmed past, or huge obs forced early compression.',
    },
    primary_failure_cause: {
      type: 'string',
      enum: ['compression-info-loss', 'summary-anchored-wrong', 'topk-overwhelm', 'never-retrieved-answer', 'had-answer-didnt-use', 'bad-search-strategy', 'wrong-finish-format', 'context-cap-nofinish', 'other'],
    },
    answer_reachable: { type: 'string', enum: ['yes', 'no', 'unclear'] },
    fix_lever: { type: 'string', enum: ['compression-design', 'retrieval-budget', 'prompting', 'training', 'tooling', 'none'] },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    one_line: { type: 'string', description: 'Specific diagnosis citing the sub-traj index where it went wrong.' },
  },
  required: ['point', 'bucket', 'gold_really_present_early', 'compression_loss', 'summary_quality', 'topk_overwhelm', 'primary_failure_cause', 'fix_lever', 'one_line'],
}

// args = { points: [ { name, files:[...] } | { name, dir, count } ] }
function _pad4(k) { let s = '' + k; while (s.length < 4) s = '0' + s; return s }
const points = (args && args.points) || []
const jobs = []
for (const p of points) {
  let files = p.files || []
  if ((!files || !files.length) && p.dir && p.count) {
    files = []
    for (let k = 0; k < p.count; k++) files.push(`${p.dir}/chain_${_pad4(k)}.json`)
  }
  for (const f of files) jobs.push({ point: p.name, path: f })
}
log(`Compression-review: ${jobs.length} failure chains across ${points.length} point(s)`)

phase('Diagnose')
const verdicts = (await parallel(jobs.map((j, i) => () =>
  agent(
    `You are diagnosing ONE failed BrowseComp-Plus agentic-search rollout to determine whether ` +
    `CONTEXT COMPRESSION caused the failure. Background: this agent runs long-horizon search; when ` +
    `its context hits ~85% full it must STOP, write a <summary> handover, and start a FRESH ` +
    `sub-trajectory that sees ONLY that summary (all prior searches/observations are discarded). So ` +
    `if an early sub-traj retrieved the answer but the summary didn't capture it, the answer is lost ` +
    `forever. The model can also set search topk up to 20 (default 10); big result sets bloat context.\n\n` +
    `Load the chain: \`python3 -c "import json;d=json.load(open('${j.path}'));` +
    `print('Q:',d['question']);print('GOLD:',repr(d['gold_answer']));print('FINAL_ANSWER:',repr(d['final_answer']));` +
    `print('bucket:',d['bucket'],'n_sub_trajs:',d['n_sub_trajs'],'outcomes:',d['outcomes']);` +
    `print('summary_sources:',d['summary_sources']);` +
    `print('FLAGS dropped=%s summary_dropped=%s had_early=%s in_final_ctx=%s'%(d['flag_dropped_by_compression'],d['flag_summary_dropped_gold'],d['flag_had_gold_early'],d['flag_gold_in_final_ctx']));` +
    `print('gold_present_by_subtraj:',d['gold_present_by_subtraj']);print('gold_in_summary_by_subtraj:',d['gold_in_summary_by_subtraj']);` +
    `[ (print('\\n==== SUBTRAJ',s['i'],'outcome=',s['outcome'],'summary_source=',s['summary_source'],'topk=',s['topk_used'],'n_open=',s['n_open']),` +
    `print('GOLD_WINDOWS:',s['gold_windows']),print('SUMMARY_WRITTEN:',s['summary'])) for s in d['sub_trajs'] ]"\`\n` +
    `(If you need to see a sub-traj's actual search queries/reasoning, re-run printing d['sub_trajs'][K]['response'].)\n\n` +
    `Your tasks:\n` +
    `1. gold_really_present_early — the FLAGS above are from a strict substring match and may be WRONG. ` +
    `Look at GOLD_WINDOWS: was the gold answer ACTUALLY retrieved (in a search result / opened page) or ` +
    `known in an early sub-traj, or is it a coincidental substring (a common word appearing unrelated)? ` +
    `If gold_windows are empty and flags are false, the answer was likely never found.\n` +
    `2. compression_loss — THE KEY JUDGMENT. Compare where the gold appeared (GOLD_WINDOWS per sub-traj) ` +
    `against the SUMMARY_WRITTEN by that sub-traj and what the later sub-trajs had. Did a <summary> ` +
    `handover DROP the answer or a strong lead that an earlier sub-traj had found? Or did the summary ` +
    `carry a DISTORTED/wrong fact? Or was there no real loss (recovered later / cause is elsewhere)?\n` +
    `3. summary_quality — assess the handover summary(ies) most relevant to the failure.\n` +
    `4. topk_overwhelm — did large observations (high topk, many docs) bury the answer or force early compression?\n` +
    `5. primary_failure_cause + answer_reachable + fix_lever + one_line (cite the sub-traj index).\n` +
    `Be skeptical and specific. Return the structured verdict; set point="${j.point}", and set bucket + rollout_id from the file's own 'bucket' and 'rollout_id' fields.`,
    { label: `cmp:${j.point}:${i}`, phase: 'Diagnose', schema: VERDICT, agentType: 'Explore' }
  ).then(v => v ? { ...v, point: j.point } : null)
))).filter(Boolean)

log(`Compression-review: ${verdicts.length} verdicts; synthesizing`)

phase('Synthesize')
const compact = verdicts.map(v => ({
  point: v.point, bucket: v.bucket, rollout_id: v.rollout_id,
  gold_early: v.gold_really_present_early, loss: v.compression_loss,
  summary_quality: v.summary_quality, topk: v.topk_overwhelm,
  cause: v.primary_failure_cause, reachable: v.answer_reachable,
  lever: v.fix_lever, one_line: v.one_line,
}))
const taxonomy = await agent(
  `You are synthesizing a study of whether CONTEXT COMPRESSION causes failures in a Qwen3.5-4B ` +
  `SUPO/BrowseComp-Plus long-horizon search agent, across 4 checkpoints (base, iter04, iter24, iter44). ` +
  `Each verdict is one failed rollout that involved >=1 compression. Chains were sampled into buckets: ` +
  `'dropped' (objective flag: gold retrieved early then not in final context), 'summary_lossy' (summary ` +
  `omitted a retrieved gold but maybe recovered), 'hi_topk' (used topk>=20), 'control' (a compression-failure ` +
  `with NO objective flag — used to catch losses the substring match MISSED).\n\n` +
  `An objective pass already measured (lower bounds, strict substring): dropped_by_compression as a ` +
  `fraction of failures = base 13.1%, iter04 18.6%, iter24 9.0%, iter44 9.1%; failures issue higher-topk ` +
  `searches than correct ones at every checkpoint. Your job is to VALIDATE and enrich this with the agent ` +
  `judgments.\n\nWrite a markdown report:\n` +
  `1. PRECISION of the objective flag: within the 'dropped' bucket, what fraction did agents CONFIRM as a ` +
  `real compression loss (compression_loss in confirmed-answer-dropped/confirmed-lead-dropped) vs refute ` +
  `(no-false-positive / no-loss-recovered / no-not-compression)? Same for 'summary_lossy'.\n` +
  `2. RECALL: within the 'control' bucket (no objective flag), what fraction did agents nonetheless find a ` +
  `compression loss (semantic loss the substring missed)? Use this to argue the objective rate is a lower/upper bound.\n` +
  `3. compression_loss distribution overall and PER CHECKPOINT — does confirmed compression-loss fall with ` +
  `training (as the summary-quality curve and the objective dropped-rate suggest: iter04 worst -> iter44 better)?\n` +
  `4. summary_quality distribution per checkpoint — is the dominant defect omission (lossy-omitted-key-fact), ` +
  `distortion (distorted-wrong-fact), or anchoring on a wrong candidate (anchored-wrong-candidate)? How does it evolve?\n` +
  `5. top_k overwhelm: how many failures show mild/severe overwhelm, and does it co-occur with compression loss ` +
  `(big obs -> faster compression -> drop)? Cite the hi_topk bucket.\n` +
  `6. primary_failure_cause breakdown: of ALL these compression-involved failures, what share is truly ` +
  `compression-info-loss vs summary-anchored-wrong vs had-answer-didnt-use vs never-retrieved vs bad-search vs wrong-finish?\n` +
  `7. 6-10 concrete, cited takeaways (rollout_ids) focused on the COMPRESSION MECHANISM: what should change in ` +
  `the summary prompt / compression policy / retrieval budget (topk cap, reserve-budget-to-finish, force-carry ` +
  `answer-candidates & verified facts into every summary, etc.). Distinguish fixes that are compression-design ` +
  `vs prompting vs training.\n` +
  `Be specific and quantitative; cite rollout_ids. Return ONLY the markdown.\n\nVERDICTS:\n${JSON.stringify(compact, null, 1)}`,
  { label: 'synthesize', phase: 'Synthesize' }
)

return { n_verdicts: verdicts.length, verdicts, taxonomy_markdown: taxonomy }
