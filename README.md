# iket-profiling-skill

An [Agent Skill](https://agentskills.io) for profiling inside CuTe DSL kernels with IKET
(In-Kernel Event Tracing) and the `run-iket` profiler.

This is not new work. The skill is a condensed summary of NVIDIA's official CUTLASS guide,
[IKET Profiling](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/iket_profiling.html),
written for coding agents. The guide is the source of truth; where they differ, follow the guide.
IKET is experimental, and its API and output may change.

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Install

The skill is the [`skills/iket-profiling/`](skills/iket-profiling) directory. Install it with
the [skills](https://github.com/vercel-labs/skills) CLI:

```sh
npx skills add humanfia/iket-profiling-skill
```

or copy the directory into your agent's skills directory, for example for Claude Code:

```sh
git clone https://github.com/humanfia/iket-profiling-skill.git
cp -r iket-profiling-skill/skills/iket-profiling ~/.claude/skills/
```

The directory name must stay `iket-profiling`, the skill's `name`.

## Usage

The agent loads the skill when you develop or tune a CuTe DSL (`cutlass.cute`) kernel and ask
where its time goes inside the kernel: phase timelines, TMA/MMA activity, or pipeline waits. It
covers the `cutlass.cute.experimental.iket` API, the `run-iket` workflow, instrumentation rules,
the Perfetto and JSON output, limitations, and troubleshooting. You need an `nvidia-cutlass-dsl`
installation that includes `run-iket` and an SM90 or newer GPU.

## Maintainers

[@DongyunZou](https://github.com/DongyunZou)

## Contributing

Issues and pull requests are welcome; see the
[contributing guide](https://github.com/humanfia/.github/blob/main/CONTRIBUTING.md). Every
statement in the skill must be backed by the official guide. Check changes with
[skills-ref](https://github.com/agentskills/agentskills/tree/main/skills-ref):

```sh
uvx --from "git+https://github.com/agentskills/agentskills#subdirectory=skills-ref" \
  skills-ref validate skills/iket-profiling
```

## License

[Apache-2.0](LICENSE) © Humanfia, for the text of this repository. The skill summarises and
adapts NVIDIA's CUTLASS documentation, which is © NVIDIA Corporation under the BSD-3-Clause
license; see [NOTICE](NOTICE). IKET, CuTe DSL, and CUTLASS are NVIDIA projects.
