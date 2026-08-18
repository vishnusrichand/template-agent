# Changelog

## [1.1.0](https://github.com/vishnusrichand/template-agent/compare/v1.0.0...v1.1.0) (2026-08-18)


### Features

* add /version endpoint and bump to 0.2.0 ([#89](https://github.com/vishnusrichand/template-agent/issues/89)) ([662778f](https://github.com/vishnusrichand/template-agent/commit/662778f600445a2a612c3131c73b2394a5ee2414))
* Add api_key authentication method for mcp ([#95](https://github.com/vishnusrichand/template-agent/issues/95)) ([6c1ab66](https://github.com/vishnusrichand/template-agent/commit/6c1ab663c8bd0d38918a02a6bfd6831270f93544))
* add custom CA certificate support at container startup ([#107](https://github.com/vishnusrichand/template-agent/issues/107)) ([3485904](https://github.com/vishnusrichand/template-agent/commit/3485904e956a321d30ab952dcac85b82f5ab1c69))
* add graceful SIGTERM shutdown with drain and resource cleanup ([9262199](https://github.com/vishnusrichand/template-agent/commit/926219939f2e92eb84748bbccd185f74f3a268b7))
* add graceful SIGTERM shutdown with drain and resource cleanup ([2549b88](https://github.com/vishnusrichand/template-agent/commit/2549b88bc9d61ca72d52c5e268887677754ed697))
* add Granite Guardian content safety guardrails ([#109](https://github.com/vishnusrichand/template-agent/issues/109)) ([d3917bb](https://github.com/vishnusrichand/template-agent/commit/d3917bbf212743a96a7ceab03af3f4a1da132c0a))
* add OpenTelemetry observability with direct OTLP export ([#83](https://github.com/vishnusrichand/template-agent/issues/83)) ([5bcddcf](https://github.com/vishnusrichand/template-agent/commit/5bcddcff69c323c39e445b474db2656f332c781d))
* add per-MCP OAuth/DCR support with token store and HTTP routes ([#78](https://github.com/vishnusrichand/template-agent/issues/78)) ([3613204](https://github.com/vishnusrichand/template-agent/commit/3613204b41cda8ecb257f60d8de20a358eb47082))
* add PII detection and scrubbing middleware ([#117](https://github.com/vishnusrichand/template-agent/issues/117)) ([d4a0f2d](https://github.com/vishnusrichand/template-agent/commit/d4a0f2d46c37cc1c01a41c9bab9c106b4d7c2957))
* add vulnerability scanning to base image build pipeline ([#100](https://github.com/vishnusrichand/template-agent/issues/100)) ([77f773e](https://github.com/vishnusrichand/template-agent/commit/77f773e14014da46d03f9bf18f1edebae494bc22))
* Adds per-thread and per-user daily LLM token tracking with MongoDB persistence and OTEL metrics export ([#76](https://github.com/vishnusrichand/template-agent/issues/76)) ([1f12507](https://github.com/vishnusrichand/template-agent/commit/1f12507de846a7ba77b60cef41a86a5b19bf8e5c))
* **agent:** mcp oauth dcr agent ([#103](https://github.com/vishnusrichand/template-agent/issues/103)) ([a6df3ea](https://github.com/vishnusrichand/template-agent/commit/a6df3eadf7444576e447f9fce503b45d6a896fd0))
* harden template-agent for production deployment ([#93](https://github.com/vishnusrichand/template-agent/issues/93)) ([42c43da](https://github.com/vishnusrichand/template-agent/commit/42c43da58a73c1a1274c61d5cb434a761cb15262))
* human in the loop ([#84](https://github.com/vishnusrichand/template-agent/issues/84)) ([1eaee38](https://github.com/vishnusrichand/template-agent/commit/1eaee38a138e2d1ff1c9f27e0d3fa360b05094ba))
* Implement Deep Agent Architecture with Orchestrator and Subagent System    ([#47](https://github.com/vishnusrichand/template-agent/issues/47)) ([51b8f3c](https://github.com/vishnusrichand/template-agent/commit/51b8f3c58e67896845007bab990579bff6110eaa))
* MCP Prefix Tool Name ([#124](https://github.com/vishnusrichand/template-agent/issues/124)) ([3a317a2](https://github.com/vishnusrichand/template-agent/commit/3a317a2cd039444397a507c7dc5b3220bc72799a))
* MCP server and LLM provider health checks with OTEL gauges ([#86](https://github.com/vishnusrichand/template-agent/issues/86)) ([974c3dd](https://github.com/vishnusrichand/template-agent/commit/974c3dd441c4c5378d2bb3ebc7562b149f550639))
* **mcp:** add DCR OAuth authentication flow for MCP tool servers ([#97](https://github.com/vishnusrichand/template-agent/issues/97)) ([e0d0003](https://github.com/vishnusrichand/template-agent/commit/e0d000321424b40464bd24970f97f60a450c3a1f))
* productionize agent — dead code removal, middleware fix, observability cleanup ([#57](https://github.com/vishnusrichand/template-agent/issues/57)) ([d5994d8](https://github.com/vishnusrichand/template-agent/commit/d5994d8524de62909bc312d9ee4124198b830be0))
* unify trace_id propagation across OTEL, Langfuse, and token budget ([#88](https://github.com/vishnusrichand/template-agent/issues/88)) ([29561be](https://github.com/vishnusrichand/template-agent/commit/29561beb2ae8146ce0e3b0fc94f05360e58f2da3))
* X-Request-ID propagation with org_id and agent_id log binding ([#87](https://github.com/vishnusrichand/template-agent/issues/87)) ([881803a](https://github.com/vishnusrichand/template-agent/commit/881803a510c4a049b4287af6d47f638c0e409389))


### Bug Fixes

* add postgres and redis components to OpenShift overlay ([#155](https://github.com/vishnusrichand/template-agent/issues/155)) ([0aa2581](https://github.com/vishnusrichand/template-agent/commit/0aa2581db99e9b3d9719645b661322930bc149ec))
* add PVC reclaimPolicy and fix issues for openshift manifest for postgres and redis ([#159](https://github.com/vishnusrichand/template-agent/issues/159)) ([395fb58](https://github.com/vishnusrichand/template-agent/commit/395fb581ab5435e42f3f53a4b978623bbce2b923))
* add token usage feature flag ([#137](https://github.com/vishnusrichand/template-agent/issues/137)) ([d90405b](https://github.com/vishnusrichand/template-agent/commit/d90405b5066ccecae40d97e0d1831df2aa0e2171))
* **agent:** expose MCP tools when servers are declared without explicit tool list ([#59](https://github.com/vishnusrichand/template-agent/issues/59)) ([cdaf2c4](https://github.com/vishnusrichand/template-agent/commit/cdaf2c49f389297b00dd8a54b5b3e9efe76d1379))
* bump mcp to 1.28.1 to fix CVE-2026-59950 ([#122](https://github.com/vishnusrichand/template-agent/issues/122)) ([00d0cca](https://github.com/vishnusrichand/template-agent/commit/00d0cca4c9b0707261de03574565ba7eef9011b8))
* bump pytest-asyncio from 1.0.0 to 1.4.0 ([#114](https://github.com/vishnusrichand/template-agent/issues/114)) ([c486a32](https://github.com/vishnusrichand/template-agent/commit/c486a3204d5e8ace0a2b9de5af1b84c85e479856))
* Fixed skill reading in deep agent ([#94](https://github.com/vishnusrichand/template-agent/issues/94)) ([6ffff54](https://github.com/vishnusrichand/template-agent/commit/6ffff542474e97c328b6a35b24219f4be334df60))
* guard atexit double-registration, validate timeout budget, set AEGRA_CONFIG ([ef442ad](https://github.com/vishnusrichand/template-agent/commit/ef442adbf00060b2540181ca5520f67c785bb441))
* headless agent deployment ([#148](https://github.com/vishnusrichand/template-agent/issues/148)) ([fc0b8cc](https://github.com/vishnusrichand/template-agent/commit/fc0b8ccd4b93606c4a8baedb1ccd13317eafcbfc))
* inherit model and MCP tools for subagents missing frontmatter fields ([#62](https://github.com/vishnusrichand/template-agent/issues/62)) ([1530254](https://github.com/vishnusrichand/template-agent/commit/1530254cc9a3a49975637dede8fcb315b60ccf4c))
* make /app group-writable for OpenShift CA cert support  Fixes [#128](https://github.com/vishnusrichand/template-agent/issues/128) ([#127](https://github.com/vishnusrichand/template-agent/issues/127)) ([0b3d873](https://github.com/vishnusrichand/template-agent/commit/0b3d87307ce571110d8ff456ae7d62296eda1ec9))
* make shutdown visible in container logs ([73bccfb](https://github.com/vishnusrichand/template-agent/commit/73bccfb278dcd371d50a7e2966735b61567c7529))
* prevent blocking for human-in-loop feature for subagent internal tools ([#91](https://github.com/vishnusrichand/template-agent/issues/91)) ([2637a29](https://github.com/vishnusrichand/template-agent/commit/2637a2976feb01a0f8d8833d37a761780c761186))
* removed unused config files ([#74](https://github.com/vishnusrichand/template-agent/issues/74)) ([b81f122](https://github.com/vishnusrichand/template-agent/commit/b81f12244671e331dbf4d8703d36ff7a67ae47b2))
* resolve pre-commit failures (ruff, mypy, pydocstyle, formatting) ([#96](https://github.com/vishnusrichand/template-agent/issues/96)) ([62eacbf](https://github.com/vishnusrichand/template-agent/commit/62eacbf0c15c0916e604a83d50a8fd59cb88c775))
* set defualt human in the loop setting to true ([#115](https://github.com/vishnusrichand/template-agent/issues/115)) ([581f8b7](https://github.com/vishnusrichand/template-agent/commit/581f8b7a12bcb0c875171588c32fe6e52538a1a0))
* set imagePullPolicy to IfNotPresent for Kind overlay ([#153](https://github.com/vishnusrichand/template-agent/issues/153)) ([62b7a59](https://github.com/vishnusrichand/template-agent/commit/62b7a599b23382c545334e227d42f68d5db89890))
* skip aegra db_manager initialization in in-memory mode ([#60](https://github.com/vishnusrichand/template-agent/issues/60)) ([918047b](https://github.com/vishnusrichand/template-agent/commit/918047b0c9afb625f869862dd566d3e6fb14c38a))
* subagent backend context bug ([#146](https://github.com/vishnusrichand/template-agent/issues/146)) ([a159777](https://github.com/vishnusrichand/template-agent/commit/a1597776ab87e71b28214fe6e658711cdc6f7338))
* subagents inherit model and MCPs from parent orchestrator ([65d8703](https://github.com/vishnusrichand/template-agent/commit/65d8703698f2335f637b50b0160052909ea452cf))
* subagents inherit model and MCPs from parent orchestrator ([5b21754](https://github.com/vishnusrichand/template-agent/commit/5b2175401c9886c4bc5f4f1bb3feb7adb9cd7168))
* Support model corps model ([#167](https://github.com/vishnusrichand/template-agent/issues/167)) ([bf3948b](https://github.com/vishnusrichand/template-agent/commit/bf3948b8b549200174e83f1952a2f36fe6241221))
* update compose networking and profiles for observability ([#123](https://github.com/vishnusrichand/template-agent/issues/123)) ([a7400d9](https://github.com/vishnusrichand/template-agent/commit/a7400d9e9b4ec70f0bc78b3235da125331ced7e7))
* use atexit as primary shutdown path, signal handlers as upgrade ([afa5b74](https://github.com/vishnusrichand/template-agent/commit/afa5b74791c87d0ff6f32695a304016bbabd0870))

## Changelog

All notable changes to this project will be documented in this file.

This changelog is automatically maintained by [release-please](https://github.com/googleapis/release-please).
