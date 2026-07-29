export const meta = {
  name: 'bcplus-correct-review',
  description: 'Review CORRECT BrowseComp-Plus eval trajectories for latent/process problems, then synthesize',
  phases: [
    { title: 'Review', detail: 'one read-only agent per correct trajectory' },
    { title: 'Synthesize', detail: 'aggregate into a process-quality taxonomy' },
  ],
}

// args = { points: [ { name, dir, count } | { name, files:[...] } ] }
// Each traj JSON (sandbox-readable under /home) has: question, gold_answer,
// model_answer, trajectory (full multi-turn text), outcome, n_turns_used,
// n_search, n_open, score (these are all CORRECT rollouts, score>=1).

const VERDICT = {
  type: 'object',
  additionalProperties: false,
  properties: {
    rollout_id: { type: ['integer', 'string', 'null'] },
    point: { type: 'string' },
    process_quality: { type: 'string', enum: ['clean', 'lucky-or-ungrounded', 'flawed-but-correct', 'inefficient', 'format-fragile'] },
    answer_grounded: { type: 'string', enum: ['yes', 'partial', 'no'], description: 'was the final answer actually supported by retrieved/opened evidence in the trajectory' },
    latent_issue: { type: 'string', description: 'short description of the latent problem, or "none" if clean' },
    fix_lever: { type: 'string', enum: ['prompting', 'training', 'tooling', 'none'] },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    one_line: { type: 'string' },
  },
  required: ['point', 'process_quality', 'answer_grounded', 'latent_issue', 'fix_lever', 'one_line'],
}

function _pad4(k) { let s = '' + k; while (s.length < 4) s = '0' + s; return s }

const points = (args && args.points) || []
const jobs = []
for (const p of points) {
  let files = p.files || []
  if ((!files || !files.length) && p.dir && p.count) {
    files = []
    for (let k = 0; k < p.count; k++) files.push(`${p.dir}/traj_${_pad4(k)}.json`)
  }
  for (const f of files) jobs.push({ point: p.name, path: f })
}
log(`Correct-review: scanning ${jobs.length} CORRECT trajectories across ${points.length} point(s)`)

phase('Review')
const verdicts = (await parallel(jobs.map((j, i) => () =>
  agent(
    `You are QUICKLY reviewing ONE BrowseComp-Plus agentic-search trajectory that got the CORRECT ` +
    `answer (the judge scored it right). Your job is NOT to re-judge correctness — it's to spot ` +
    `LATENT / process problems where the right answer came via a shaky path.\n\n` +
    `Read the JSON file at: ${j.path} (use Bash: ` +
    `\`python3 -c "import json;d=json.load(open('${j.path}'));print('Q:',d['question']);` +
    `print('GOLD:',d['gold_answer']);print('MODEL:',d['model_answer']);print('n_search:',d['n_search'],` +
    `'n_open:',d['n_open'],'turns:',d['n_turns_used']);print(d['trajectory'])"\`).\n\n` +
    `Assess (be quick, this is a skim):\n` +
    `1. answer_grounded — is the final answer actually SUPPORTED by evidence the agent retrieved/opened ` +
    `in this trajectory, or did it come from a guess / prior parametric knowledge / an unverified ` +
    `handover summary / a hallucinated detail that happened to match gold?\n` +
    `2. process_quality — clean (grounded, efficient) / lucky-or-ungrounded (right answer, no real ` +
    `evidence or lucky guess) / flawed-but-correct (right answer via wrong or sloppy reasoning, misread ` +
    `a doc but still got it, or the judge was lenient on a borderline match) / inefficient (many wasted ` +
    `searches, near the context cap) / format-fragile (answer barely matches gold, could easily be ` +
    `marked wrong).\n` +
    `3. latent_issue — one short phrase, or "none" if genuinely clean.\n` +
    `Return the structured verdict. Set point="${j.point}" and rollout_id from the file if present.`,
    { label: `chk:${j.point}:${i}`, phase: 'Review', schema: VERDICT, agentType: 'Explore' }
  ).then(v => v ? { ...v, point: j.point } : null)
))).filter(Boolean)

log(`Correct-review: ${verdicts.length} verdicts; synthesizing`)

phase('Synthesize')
const compact = verdicts.map(v => ({
  point: v.point, quality: v.process_quality, grounded: v.answer_grounded,
  lever: v.fix_lever, issue: v.latent_issue, one_line: v.one_line, rollout_id: v.rollout_id,
}))
const taxonomy = await agent(
  `You are synthesizing a PROCESS-QUALITY review of CORRECT BrowseComp-Plus trajectories for a ` +
  `Qwen3.5-4B SUPO/BC+ run, across several checkpoints. Below is a JSON array of per-trajectory ` +
  `verdicts (all were scored correct). Write a markdown report:\n` +
  `1. process_quality distribution overall and PER CHECKPOINT (base/iter04/iter14/iter24/iter34/iter44 ` +
  `if present) — does the fraction of clean vs lucky-or-ungrounded/flawed-but-correct change with training?\n` +
  `2. answer_grounded breakdown (yes/partial/no) per checkpoint — what % of CORRECT answers were NOT ` +
  `grounded in retrieved evidence (i.e. lucky/parametric/handover-trusted)? This is the key number: ` +
  `"how many wins are fragile".\n` +
  `3. The most common latent issues, with 3-5 representative rollout_ids each.\n` +
  `4. Cross-cut with the failure taxonomy: which latent issues in CORRECT trajectories mirror the actual ` +
  `FAILURE modes (e.g. ungrounded/handover-trusted answers that happened to be right vs wrong)?\n` +
  `5. 5-8 concrete takeaways: which "wins" are real vs luck, and what it implies for the eval metric and training.\n` +
  `Be specific, cite rollout_ids. Return ONLY the markdown.\n\nVERDICTS:\n${JSON.stringify(compact, null, 1)}`,
  { label: 'synthesize', phase: 'Synthesize' }
)

return { n_verdicts: verdicts.length, verdicts, taxonomy_markdown: taxonomy }
