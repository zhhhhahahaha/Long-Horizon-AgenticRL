export const meta = {
  name: 'bcplus-summary-retention',
  description: 'Judge whether the FINAL handover summary retained an answer the model had already retrieved earlier (summary content ability)',
  phases: [
    { title: 'Judge', detail: 'one agent per candidate failure' },
    { title: 'Synthesize', detail: 'per-checkpoint summary-retention rate' },
  ],
}

// args = { points: [ { name, dir, count } ] }  (candidates pre-filtered part-wise: failure, >=2 subtraj,
// gold retrieved in a NON-final sub-traj's observations). Each cand JSON has: question, gold_answer,
// final_answer, n_sub_trajs, outcomes, sub_trajs[] {i, is_final, obs_gold_windows (raw retrieval snippets
// around gold, non-final only), summary (full handover text), response (truncated)}.

const VERDICT = {
  type: 'object',
  additionalProperties: false,
  properties: {
    rollout_id: { type: ['integer', 'string', 'null'] },
    point: { type: 'string' },
    retrieved_earlier: {
      type: 'string',
      enum: ['in-search-result', 'in-opened-page', 'in-reasoning-only', 'not-really'],
      description: 'Did the model ACTUALLY retrieve the gold answer in a NON-final sub-traj? Confirm the pre-filter: look at obs_gold_windows. in-search-result / in-opened-page = genuinely retrieved; in-reasoning-only = it only appears in the model text not retrieval; not-really = the part-wise match was coincidental (parts scattered, not the actual answer).',
    },
    final_summary_retained: {
      type: 'string',
      enum: ['carried', 'dropped', 'distorted', 'na-not-retrieved'],
      description: 'Look at the FINAL handover summary (the summary written by the second-to-last sub-traj, which fed the last sub-traj). Did it carry the answer or its critical identifying fact/lead forward? carried = yes (even if paraphrased); dropped = the answer/lead is simply absent; distorted = present but wrong/misleading; na-not-retrieved = set only if retrieved_earlier=not-really.',
    },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    one_line: { type: 'string', description: 'Specific: what was retrieved, and whether the final summary kept it.' },
  },
  required: ['point', 'retrieved_earlier', 'final_summary_retained', 'one_line'],
}

function _pad4(k){let s=''+k;while(s.length<4)s='0'+s;return s}
const points = (args && args.points) || []
const jobs = []
for (const p of points) {
  let files = p.files || []
  if ((!files || !files.length) && p.dir && p.count)
    for (let k=0;k<p.count;k++) files.push(`${p.dir}/cand_${_pad4(k)}.json`)
  for (const f of files) jobs.push({ point: p.name, path: f })
}
log(`Summary-retention: ${jobs.length} candidate failures across ${points.length} point(s)`)

phase('Judge')
const verdicts = (await parallel(jobs.map((j,i)=>()=>
  agent(
    `You are judging ONE failed BrowseComp-Plus search rollout for a SPECIFIC question: when the model ` +
    `had ALREADY retrieved the answer in an earlier session, did its FINAL handover summary keep it?\n\n` +
    `Background: this long-horizon agent compresses when context fills — it writes a <summary>, then a ` +
    `fresh sub-trajectory continues from that summary ALONE (all prior searches discarded). So a summary ` +
    `that omits an already-found answer loses it permanently. This rollout was PRE-FILTERED (lenient) as ` +
    `one where the gold answer plausibly appeared in an early sub-traj's retrieval — your job is to confirm ` +
    `that and judge whether the FINAL summary retained it.\n\n` +
    `Load: \`python3 -c "import json;d=json.load(open('${j.path}'));print('Q:',d['question']);` +
    `print('GOLD:',repr(d['gold_answer']));print('FINAL_ANSWER:',repr(d['final_answer']));` +
    `print('n_sub_trajs:',d['n_sub_trajs'],'outcomes:',d['outcomes']);` +
    `[ (print('\\n==SUBTRAJ',s['i'],'is_final=',s['is_final'],'outcome=',s['outcome']),` +
    `print('OBS_GOLD_WINDOWS:',s['obs_gold_windows']),print('SUMMARY_WRITTEN:',s['summary'])) for s in d['sub_trajs'] ]"\`\n` +
    `(obs_gold_windows = raw retrieval snippets around the gold, shown for NON-final sub-trajs. The FINAL ` +
    `handover summary = the 'SUMMARY_WRITTEN' of the SECOND-TO-LAST sub-traj. Re-print d['sub_trajs'][K]['response'] if needed.)\n\n` +
    `Judge two things:\n` +
    `1. retrieved_earlier — from obs_gold_windows, did the model GENUINELY retrieve the gold answer in a ` +
    `non-final sub-traj (in a search result or opened page), or is the pre-filter a coincidence (parts ` +
    `scattered, not the real answer)? Be strict here — this guards against false positives.\n` +
    `2. final_summary_retained — read the FINAL handover summary; did it carry the answer / its critical ` +
    `identifying fact forward (carried, even if reworded), or drop it, or distort it? (semantic judgment — ` +
    `a paraphrase that unambiguously names the answer counts as carried.)\n` +
    `Return the verdict; set point="${j.point}", rollout_id from the file.`,
    { label: `sr:${j.point}:${i}`, phase: 'Judge', schema: VERDICT, agentType: 'Explore' }
  ).then(v => v ? {...v, point: j.point} : null)
))).filter(Boolean)

log(`Summary-retention: ${verdicts.length} verdicts; synthesizing`)

phase('Synthesize')
const compact = verdicts.map(v=>({point:v.point, retrieved:v.retrieved_earlier, retained:v.final_summary_retained, rollout_id:v.rollout_id, one_line:v.one_line}))
const taxonomy = await agent(
  `Synthesize a summary-content-retention study for a Qwen3.5-4B SUPO/BrowseComp-Plus agent across 4 ` +
  `checkpoints (base, iter04, iter24, iter44). Each verdict is a FAILED rollout pre-filtered as "gold ` +
  `plausibly retrieved in an early sub-traj"; an agent judged (1) retrieved_earlier (genuine retrieval vs ` +
  `false-positive) and (2) final_summary_retained (carried/dropped/distorted).\n\n` +
  `Write markdown:\n` +
  `1. Restrict to GENUINELY-retrieved cases (retrieved_earlier in in-search-result/in-opened-page). Per ` +
  `checkpoint, report: n_genuine, and the SUMMARY-LOSS rate = (dropped+distorted)/n_genuine, plus carried rate.\n` +
  `2. Is the loss rate FLAT across base->iter44 (i.e. training did NOT improve summary content retention), ` +
  `or does it fall? Give the exact per-checkpoint numbers and state the trend.\n` +
  `3. false-positive rate of the pre-filter (not-really / in-reasoning-only share) per checkpoint.\n` +
  `4. 4-6 concrete cited examples (rollout_ids) of "retrieved in search, dropped by final summary".\n` +
  `5. One-paragraph conclusion for a report: does RL training fix the model's summary CONTENT ability?\n` +
  `Be quantitative; cite rollout_ids. Return ONLY markdown.\n\nVERDICTS:\n${JSON.stringify(compact,null,1)}`,
  { label: 'synthesize', phase: 'Synthesize' }
)
return { n_verdicts: verdicts.length, verdicts, taxonomy_markdown: taxonomy }
