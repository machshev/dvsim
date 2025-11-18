# CHANGELOG

<!-- version list -->

## v1.3.0 (2025-11-18)

### Features

- Add block report template
  ([`6e716fe`](https://github.com/lowRISC/dvsim/commit/6e716fea6f29c7a976d6f876fe6b6830827ae785))

- Add jinja2 template renderer
  ([`efeb68a`](https://github.com/lowRISC/dvsim/commit/efeb68add7b2b59e2b28cbbee0bdde04d648b5f0))

- Add report generation from templates
  ([`14406e4`](https://github.com/lowRISC/dvsim/commit/14406e40f059227bbdf00f6ee9134b23a612ca7f))

- Add summary report template
  ([`fa0852a`](https://github.com/lowRISC/dvsim/commit/fa0852aff97dd9e9a892c73f38b313e04e92a186))

- Redirect template
  ([`02b05fe`](https://github.com/lowRISC/dvsim/commit/02b05fe82ac0090f3b4a9cd708615150bfb62afd))

### Refactoring

- Clean up unused functions
  ([`40958fd`](https://github.com/lowRISC/dvsim/commit/40958fdbf325bd90eb75b27234759c6b6e003d0c))

- Tidy up results generation with direct model creation
  ([`75d91a3`](https://github.com/lowRISC/dvsim/commit/75d91a3cb37ff918c61f636748992265c29ffd68))


## v1.2.0 (2025-11-14)

### Features

- Add JSON summary generation
  ([`701cf04`](https://github.com/lowRISC/dvsim/commit/701cf048d2feca80e43b84b3b9c75b271bbdc0ea))

- Add ResultsSummary model
  ([`2c5b1e9`](https://github.com/lowRISC/dvsim/commit/2c5b1e95d73b01536ad802aa91d6ada819395513))


## v1.1.0 (2025-11-13)

### Features

- Add SimTool interface and implementations
  ([`b649826`](https://github.com/lowRISC/dvsim/commit/b64982650a104c69106e9d383144182b57b7bf15))

### Refactoring

- Use the tool plugins directly
  ([`b6416fa`](https://github.com/lowRISC/dvsim/commit/b6416fab1e88396d6f0244f4c77b7c780fba1043))

### Testing

- Add initial tests for the VCS tool plugin.
  ([`21edab1`](https://github.com/lowRISC/dvsim/commit/21edab12ec38dc586088d08e1d7f24dbcc0332c3))

- Add tests for the tool plugin system
  ([`b883b7b`](https://github.com/lowRISC/dvsim/commit/b883b7b73688f0cdde6bd50cfcbb98b07c2d177d))


## v1.0.6 (2025-11-12)

### Bug Fixes

- Report item filtering
  ([`4b5d01d`](https://github.com/lowRISC/dvsim/commit/4b5d01d761454e82e21cd34529eeb8af5ed9a4a6))

- Run and sim time precision and units
  ([`8af1207`](https://github.com/lowRISC/dvsim/commit/8af12074f27e3f6e24694691384be8b5381908a9))


## v1.0.5 (2025-11-11)

### Chores

- Nix flake update
  ([`80545d3`](https://github.com/lowRISC/dvsim/commit/80545d3c4c1f131403cb17b9fc485adf0ddb7e56))

### Refactoring

- Add JobSpec common abstraction
  ([`be1e1e1`](https://github.com/lowRISC/dvsim/commit/be1e1e1c56eb3842e75a1c6352b484a1b756fb6a))

- Migrate from Depoy.dump to JobSpec.model_dump
  ([`91ac90e`](https://github.com/lowRISC/dvsim/commit/91ac90e5a1f24f2cbeda9d1811bc95eb6b226547))


## v1.0.4 (2025-11-06)

### Code Style

- Linting, docstrings and typing
  ([`08b8e6d`](https://github.com/lowRISC/dvsim/commit/08b8e6de9827d5f2a2e7d4c493d76f097d5f4ace))

### Refactoring

- Add WorkspaceCfg
  ([`9a7a08e`](https://github.com/lowRISC/dvsim/commit/9a7a08e57d53bbf24f7b414ef84aee31c2d2b131))

- Improvements in lsf launcher
  ([`d64033d`](https://github.com/lowRISC/dvsim/commit/d64033d580b7acaba61b9fc98515e22de720dfe5))

- Make cov_db_dirs deterministic
  ([`3bb06fc`](https://github.com/lowRISC/dvsim/commit/3bb06fc14fb041a253da4d1af5ffc9989e2e63b3))

- Rename model_dump -> dump
  ([`53bbfd4`](https://github.com/lowRISC/dvsim/commit/53bbfd41ee38b7f0fcf97e07a7520573077db585))

### Testing

- Add initial CompileSim unittest
  ([`e8d5279`](https://github.com/lowRISC/dvsim/commit/e8d52791c2ac5317648cb51f495db4eb8aeca3e7))


## v1.0.3 (2025-10-30)

### Bug Fixes

- Add missing concrete implementations
  ([`49786d0`](https://github.com/lowRISC/dvsim/commit/49786d0005a35215059d540b7d9136ae9d8b1ad5))

- Remove dependency on launcher
  ([`c135fa6`](https://github.com/lowRISC/dvsim/commit/c135fa6562132cf3337a6abb17942ff361d8fe5b))

### Code Style

- Improved docstrings and linting fixes
  ([`bc1cdef`](https://github.com/lowRISC/dvsim/commit/bc1cdef82e8c8fde84ca9c10aba5731d108989d7))


## v1.0.2 (2025-10-16)

### Bug Fixes

- Remove use of feature not supported by 3.10
  ([`b8f45ef`](https://github.com/lowRISC/dvsim/commit/b8f45ef23040f961c391d94add0275b66a91b4cb))

### Chores

- Flake update
  ([`7781364`](https://github.com/lowRISC/dvsim/commit/7781364fab1980b1448b0d8a49c6b3244af9d10a))

### Continuous Integration

- Fix python matrix
  ([`a195684`](https://github.com/lowRISC/dvsim/commit/a195684ae8d2bd6733b0cad95f59e99d182b0257))


## v1.0.1 (2025-10-15)

### Bug Fixes

- Fake launcher missing abstract methods
  ([`a58fd05`](https://github.com/lowRISC/dvsim/commit/a58fd05402439a31b443fe6b8e51dc23500bc05f))

### Refactoring

- Use deployment name instead of object as dict keys
  ([`6938d34`](https://github.com/lowRISC/dvsim/commit/6938d343d9ef44cf23b333fdb066f39d6d34973c))


## v1.0.0 (2025-10-14)

### Bug Fixes

- [launcher] drop poll_freq from 1s to 0.05s for the local launcher
  ([`3628d69`](https://github.com/lowRISC/dvsim/commit/3628d693c6beba67246c754c23f1b17013afeba4))

- [wildcards] refactor and improved testing with fixes
  ([`c7d7a9a`](https://github.com/lowRISC/dvsim/commit/c7d7a9a2292b2d1d493bc9439da78f2a42590f2a))

- Circular import issue
  ([`0a1c1c3`](https://github.com/lowRISC/dvsim/commit/0a1c1c345f8b83af21b654b14d3a8abd3fdad8d4))

- Improve testing and fix issues with the fs helpers.
  ([`40c4f22`](https://github.com/lowRISC/dvsim/commit/40c4f2241a50dc7f063fa2e9991063d04556bcf0))

- Logging of timeout after wildcard eval
  ([`87e09a3`](https://github.com/lowRISC/dvsim/commit/87e09a3e579e117361f0128d97e361daa2e245ee))

- Move ipython into debug/dev/nix extra dependency groups
  ([`7b53822`](https://github.com/lowRISC/dvsim/commit/7b53822f0f6aa3c401074853ec8580044731ecfa))

- Nix devshell
  ([`995a57c`](https://github.com/lowRISC/dvsim/commit/995a57cbb08bd4308cd6c36df1af93069ec86aef))

- Regression
  ([`9c1bf17`](https://github.com/lowRISC/dvsim/commit/9c1bf174676d0beb7898b604a3fca939b1e4ae01))

- Remove shebangs from scripts
  ([`597c7d5`](https://github.com/lowRISC/dvsim/commit/597c7d584c7dcd29c4241978c9bf274babc3c71a))

- Remove unused Bazel BUILD file
  ([`60dcb91`](https://github.com/lowRISC/dvsim/commit/60dcb91c78b4ad69e3799e94e029c0b1d2ef69be))

- Results refactor name clash
  ([`073f9a7`](https://github.com/lowRISC/dvsim/commit/073f9a7a978b021e94a794a68bbe2d034f5985d7))

- Style.css needs to be in the flow dir
  ([`a181bdd`](https://github.com/lowRISC/dvsim/commit/a181bdde42bc7e39418dc5a4c0773bdf81f8db3e))

### Build System

- [pytest] update config to ignore scratch dir when collecting tests
  ([`c484ac6`](https://github.com/lowRISC/dvsim/commit/c484ac6b023eaea91723235060e0ae2e5a99e339))

### Code Style

- [tests] disable pedantic rules for test files.
  ([`3a7709d`](https://github.com/lowRISC/dvsim/commit/3a7709da9403eee1f50762579d8a0520d6cd91c4))

- Disable TRY003
  ([`fe7fecd`](https://github.com/lowRISC/dvsim/commit/fe7fecd7bd07ef61b5410ab1ca6a09fc65d46880))

- Fix auto-fixable PTH issues
  ([`cc26b34`](https://github.com/lowRISC/dvsim/commit/cc26b342c8fa8641a0b364424bf299d34290e619))

- Fix instances of A001
  ([`bfed9d4`](https://github.com/lowRISC/dvsim/commit/bfed9d4f5f3bb535d5713d16b2eb0542d2d3ead8))

- Fix instances of N806
  ([`0cf67f1`](https://github.com/lowRISC/dvsim/commit/0cf67f149199115c9cbd111db7c5846aa331a663))

- Fix instances of N816
  ([`6e76ad7`](https://github.com/lowRISC/dvsim/commit/6e76ad704d7aaf972b54696bd268057d1ef7c976))

- Fix instances of N818
  ([`36ec821`](https://github.com/lowRISC/dvsim/commit/36ec821104de32a895b819828dcfa15530e186ae))

- Fix N803 issues and enable rule
  ([`60b16d7`](https://github.com/lowRISC/dvsim/commit/60b16d7a5966f484e75536134ef5a45f04a5e9a4))

- Remove uneccesery variable
  ([`a5290dc`](https://github.com/lowRISC/dvsim/commit/a5290dcb13475cf1da232b0ff46f4d2884b9ec9b))

### Continuous Integration

- Add an action to get a lowrisc-ci app installation access token
  ([`16a9508`](https://github.com/lowRISC/dvsim/commit/16a9508ea6954d4f3634ba8587beafe85417caa1))

- Add automated release action based on python-semantic-release / conventional commits
  ([`46dc514`](https://github.com/lowRISC/dvsim/commit/46dc514e13c10137a4704929d794ba6281a233bf))

- Copy over check for commit metadata
  ([`1dc0673`](https://github.com/lowRISC/dvsim/commit/1dc067397713edcaa56097038f716ecea6462fd2))

- Github actions to version and creation of the release
  ([`fc763d4`](https://github.com/lowRISC/dvsim/commit/fc763d4f220a6d8ad2a290bcaa639672d9c2e0cb))

### Features

- [launcher] add fake launcher to produce random results
  ([`fd5aed1`](https://github.com/lowRISC/dvsim/commit/fd5aed144be4785b48e918823198c6d548a142a7))

- Add deployment object dump debug feature
  ([`f682788`](https://github.com/lowRISC/dvsim/commit/f68278806eb4d860c30723510feb842d8a2b0efd))

- Added configuration options for release
  ([`d7ed748`](https://github.com/lowRISC/dvsim/commit/d7ed74830c83c1acc25ca60eb43144ee1cb19c19))

### Refactoring

- [flow] module rename dvsim.CdcCfg -> dvsim.flow.cdc
  ([`d93fdce`](https://github.com/lowRISC/dvsim/commit/d93fdce5942c7c0d57c4c57f8daa5a2230c3d898))

- [flow] module rename dvsim.CfgFactory -> dvsim.flow.factory
  ([`a47d9e2`](https://github.com/lowRISC/dvsim/commit/a47d9e21ef906a21902f9071484f097a37081fe0))

- [flow] module rename dvsim.FlowCfg -> dvsim.flow.base
  ([`4ec6081`](https://github.com/lowRISC/dvsim/commit/4ec60819f789e65fec0a1bcb404a875b3169becc))

- [flow] module rename dvsim.FormalCfg -> dvsim.flow.formal
  ([`c59fa69`](https://github.com/lowRISC/dvsim/commit/c59fa69d24a3c58c27386c92965ac425a6400c6f))

- [flow] module rename dvsim.LintCfg -> dvsim.flow.lint
  ([`31a5b15`](https://github.com/lowRISC/dvsim/commit/31a5b1549413dba9769ba614218b29a0793e4fe0))

- [flow] module rename dvsim.OneShotCfg -> dvsim.flow.one_shot
  ([`8ff0f09`](https://github.com/lowRISC/dvsim/commit/8ff0f0900fc3b639eb6af821213c846ce92e2cb5))

- [flow] module rename dvsim.SimCfg -> dvsim.flow.sim
  ([`4e0c39a`](https://github.com/lowRISC/dvsim/commit/4e0c39ae9bb9569edcea6b5d3ad44953bae83845))

- [flow] module rename dvsim.SynCfg -> dvsim.flow.syn
  ([`eca83a6`](https://github.com/lowRISC/dvsim/commit/eca83a6ed0e0c1ac5c408b3954e64e1d2f002cb9))

- [job] pull out JobTime tests, improved testing and fix a few bugs
  ([`b56441f`](https://github.com/lowRISC/dvsim/commit/b56441f3fe38467bdee0880251c74e4a0047c572))

- [launcher] module rename dvsim.Launcher -> dsvsim.launcher.base
  ([`f89917b`](https://github.com/lowRISC/dvsim/commit/f89917be21157d4179de1c4d22083fcbcdfedad4))

- [launcher] module rename dvsim.LauncherFactory -> dsvsim.launcher.factory
  ([`9e90ebe`](https://github.com/lowRISC/dvsim/commit/9e90ebe16e7005d02a70f7ce6247df26cae7dea5))

- [launcher] module rename dvsim.LocalLauncher -> dsvsim.launcher.local
  ([`88f8d0d`](https://github.com/lowRISC/dvsim/commit/88f8d0d348be33d0a23ca6bf5df69e29cd44f9f1))

- [launcher] module rename dvsim.LsfLauncher -> dsvsim.launcher.lsf
  ([`f2bf778`](https://github.com/lowRISC/dvsim/commit/f2bf7783066ee78ef29a5dc9638af3a61cb0fd0f))

- [launcher] module rename dvsim.NcLauncher -> dsvsim.launcher.nc
  ([`6d2806b`](https://github.com/lowRISC/dvsim/commit/6d2806b8e423832f2276bc89dba7c62f6964b7ad))

- [launcher] module rename dvsim.SgeLauncher -> dsvsim.launcher.sge
  ([`3120ec4`](https://github.com/lowRISC/dvsim/commit/3120ec408c8269454722fc6ec12f07db110b8de9))

- [launcher] module rename dvsim.SlurmLauncher -> dsvsim.launcher.slurm
  ([`0d81e22`](https://github.com/lowRISC/dvsim/commit/0d81e229cdf983f2b24d3df235d39655782fb40c))

- [logging] pull out logging setup from the main function
  ([`1e75b9a`](https://github.com/lowRISC/dvsim/commit/1e75b9a9797d3053f3db0f8e5967f53f48b49360))

- [logging] use custom logger rather than the base logger
  ([`1aa0541`](https://github.com/lowRISC/dvsim/commit/1aa0541c7b68d33bac27f7f0aeb3e4ff4696a673))

- [publish] remove the old report publishing mechanisms
  ([`c9cd75f`](https://github.com/lowRISC/dvsim/commit/c9cd75f95e74809ac530264a62a8b3a3195093d7))

- [report] remove old report dir versioning
  ([`96ff3d5`](https://github.com/lowRISC/dvsim/commit/96ff3d57938ab18942b418f75770c25650418771))

- [reporting] remove unnesesery latest dir for reporting
  ([`de0fa37`](https://github.com/lowRISC/dvsim/commit/de0fa375028dbc22fd6b97370f0111ebce33c1e0))

- [typing] add typing to the Scheduler
  ([`b796d3c`](https://github.com/lowRISC/dvsim/commit/b796d3ced10a9bf80ebd94b0070cddfc6b92e152))

- [utils] convert utils to a package
  ([`08bcbdc`](https://github.com/lowRISC/dvsim/commit/08bcbdcefb33f7cf900971e1b6ed8d3a88527ccc))

- [utils] split out remaining utils into modules
  ([`64ce14c`](https://github.com/lowRISC/dvsim/commit/64ce14cfba2074f83e76c2047ae2f39237765295))

- Improve and add typing to status printer
  ([`d097727`](https://github.com/lowRISC/dvsim/commit/d0977274057b3ccf315ddd6724f7ff85f9de4f3f))

- Initial detanglement of deployment objects
  ([`1102d8d`](https://github.com/lowRISC/dvsim/commit/1102d8ddcd4743b5e20fab571f89356c9fb914cc))

- Rename dvsim.MsgBucket -> dvsim.msg_bucket
  ([`834f9e7`](https://github.com/lowRISC/dvsim/commit/834f9e75adbe1c08687eb97c85e475d0dfca8a2d))

- Rename dvsim.MsgBuckets -> dvsim.msg_buckets
  ([`3ae5918`](https://github.com/lowRISC/dvsim/commit/3ae591851f5c201346bd6d182bd59f832c81b9ec))

- Rename dvsim.Regression -> dvsim.regression
  ([`b79b5f4`](https://github.com/lowRISC/dvsim/commit/b79b5f45009e28d9c76c99fbd956bfb023f37664))

- Rename dvsim.Scheduler -> dvsim.scheduler
  ([`afbcaa1`](https://github.com/lowRISC/dvsim/commit/afbcaa17759571c845a6e93d6fb94cf720125a93))

- Rename dvsim.SimResults -> dvsim.sim_results
  ([`b2e7813`](https://github.com/lowRISC/dvsim/commit/b2e7813e1aa1976e22414ebccb069884c8639e42))

- Rename dvsim.StatusPrinter -> dvsim.utils.status_printer
  ([`14c4917`](https://github.com/lowRISC/dvsim/commit/14c4917c9aa2a44aaaf1071e1ca5b19cf895bb89))

- Rename dvsim.Test/Testplan -> dvsim.utils.test/testplan
  ([`9b4f89f`](https://github.com/lowRISC/dvsim/commit/9b4f89fcefa43e291998d26474678e066e5e4431))

- Rename dvsim.Timer -> dvsim.utils.timer
  ([`41366ad`](https://github.com/lowRISC/dvsim/commit/41366adcf8ef3b1c433d7880e66bf0045fce3a26))

- Rename remaining modules and enable N999 lint
  ([`8dad466`](https://github.com/lowRISC/dvsim/commit/8dad4663ec4894863a0d095060f984df5cdb9cd4))

### Testing

- Add cli run test
  ([`f87b7e6`](https://github.com/lowRISC/dvsim/commit/f87b7e608820c5f1506c287ae6f0a647f561ea43))


## v0.1.0 (2025-09-09)

- Initial Release
