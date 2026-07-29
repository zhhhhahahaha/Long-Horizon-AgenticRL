export const meta = {
  name: 'bcplus-failure-deepdive',
  description: 'Per-trajectory agent diagnosis of BrowseComp-Plus eval failures, then synthesis',
  phases: [
    { title: 'Diagnose', detail: 'one read-only agent per failing trajectory' },
    { title: 'Synthesize', detail: 'aggregate verdicts into a failure taxonomy' },
  ],
}

// args = {
//   points: [ { name: "iter44", files: [...] } , ... ]        // explicit file list, OR
//   points: [ { name: "iter44", dir: "/home/.../iter44", count: 88 } ]  // dir + count
// }
// Each traj JSON file (sandbox-readable, under /home) has: question, gold_answer,
// model_answer, trajectory (full multi-turn text w/ search queries + observations),
// outcome, n_turns_used, n_search, n_open.

const VERDICT = {
  type: 'object',
  additionalProperties: false,
  properties: {
    rollout_id: { type: ['integer', 'string', 'null'] },
    point: { type: 'string' },
    failure_stage: { type: 'string', description: 'which turn/step it went wrong, e.g. "turn 4: search query too generic"' },
    failure_mode: { type: 'string', description: 'short category, e.g. bad-search-query / misread-doc / stopped-too-early / context-or-turn-limit / had-answer-but-wrong-finish / tool-or-parse-error / hallucinated / never-finished' },
    answer_reachable: { type: 'string', enum: ['yes', 'no', 'unclear'], description: 'was the gold answer reachable from what the agent retrieved / could have retrieved' },
    what_should_have_happened: { type: 'string', description: '1-3 sentences: the correct path to the answer' },
    fix_lever: { type: 'string', enum: ['prompting', 'training', 'tooling', 'genuinely-hard'] },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    one_line: { type: 'string', description: 'one-line summary of this failure' },
  },
  required: ['point', 'failure_stage', 'failure_mode', 'answer_reachable', 'what_should_have_happened', 'fix_lever', 'one_line'],
}

function _pad4(k) {
  let s = '' + k
  while (s.length < 4) s = '0' + s
  return s
}

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
log(`Stage B: diagnosing ${jobs.length} failing trajectories across ${points.length} point(s)`)

phase('Diagnose')
const verdicts = (await parallel(jobs.map((j, i) => () =>
  agent(
    `You are diagnosing ONE failed BrowseComp-Plus agentic-search trajectory.\n\n` +
    `Read the JSON file at: ${j.path}\n` +
    `It is a JSON object. The "trajectory" field is LARGE (can be ~150k chars), so read it ` +
    `robustly — e.g. via Bash: \`python3 -c "import json,textwrap;d=json.load(open('${j.path}'));` +
    `print('Q:',d['question']);print('GOLD:',d['gold_answer']);print('MODEL:',d['model_answer']);` +
    `print('OUTCOME:',d['outcome'],'turns:',d['n_turns_used'],'search:',d['n_search'],'open:',d['n_open']);` +
    `print(d['trajectory'])"\` (or the Read tool for smaller ones). The trajectory is the FULL ` +
    `multi-turn ReAct text with the model's <function=search/open_page/finish> tool calls and ` +
    `the returned observations.\n\n` +
    `This is question point "${j.point}". Analyze carefully:\n` +
    `1. Read the question and the gold answer. Understand what a correct research path looks like.\n` +
    `2. Walk the trajectory turn by turn. Identify the FIRST point where it went wrong ` +
    `(bad/overly-generic search query? retrieved the right document but misread/ignored it? ` +
    `stopped searching too early? ran out of turns/context or hit the compression cap without ` +
    `finishing? had the answer in context but emitted a wrong final answer? tool/parse error? ` +
    `hallucinated an answer?).\n` +
    `3. Judge whether the gold answer was actually REACHABLE from what it retrieved (or could ` +
    `plausibly have retrieved with better queries), vs genuinely hard/unavailable.\n` +
    `4. State concisely how it SHOULD have reached the correct answer.\n` +
    `5. Pick the fix lever: prompting / training / tooling / genuinely-hard.\n\n` +
    `Return the structured verdict. Set rollout_id from the file if present, and point="${j.point}".`,
    { label: `diag:${j.point}:${i}`, phase: 'Diagnose', schema: VERDICT, agentType: 'Explore' }
  ).then(v => v ? { ...v, rollout_id: v.rollout_id, point: j.point } : null)
))).filter(Boolean)

log(`Stage B: got ${verdicts.length} verdicts; synthesizing`)

phase('Synthesize')
// Compact the verdicts for the synthesizer's prompt.
const compact = verdicts.map(v => ({
  point: v.point,
  mode: v.failure_mode,
  reachable: v.answer_reachable,
  lever: v.fix_lever,
  stage: v.failure_stage,
  one_line: v.one_line,
  rollout_id: v.rollout_id,
}))

const taxonomy = await agent(
  `You are synthesizing a FAILURE TAXONOMY for a BrowseComp-Plus RL run (Qwen3.5-4B, SUPO/BC+).\n` +
  `Below are per-trajectory failure diagnoses (JSON array). Each has: point (checkpoint), ` +
  `failure_mode, answer_reachable, fix_lever, failure_stage, one_line, rollout_id.\n\n` +
  `Write a comprehensive markdown report with:\n` +
  `1. A failure-mode taxonomy: table of failure_mode -> count (and % ), broken down by point ` +
  `if more than one point is present (e.g. base vs iter44 shift).\n` +
  `2. Reachability breakdown (yes/no/unclear) — how often the answer WAS reachable but the ` +
  `agent still failed (these are the fixable ones).\n` +
  `3. Fix-lever breakdown (prompting/training/tooling/genuinely-hard) with the top actionable ` +
  `patterns and WHY, citing 2-4 representative rollout_ids per pattern with their one_line.\n` +
  `4. The 5-8 most important, concrete takeaways for improving the agent.\n` +
  `Be specific and cite rollout_ids. Return ONLY the markdown.\n\n` +
  `VERDICTS:\n${JSON.stringify(compact, null, 1)}`,
  { label: 'synthesize', phase: 'Synthesize' }
)

return { n_verdicts: verdicts.length, verdicts, taxonomy_markdown: taxonomy }
