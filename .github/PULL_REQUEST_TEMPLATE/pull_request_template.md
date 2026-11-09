---
name: "\U0001F41B Pull Request"
# 模板名称：拉取请求
about: Fix a bug or create new example
# 模板用途：修复缺陷或创建新示例。

---

## Description

改动说明

Please include a summary of the newly proposed example or issue being fixed. Please also include relevant motivation, context.

请概述新增示例或修复的问题，并说明相关动机和背景。

If this is a new example, how is your example different enough from the remaining examples in the repo. 

如果是新增示例，请说明它与仓库其他示例有哪些实质区别。

If this is a bug fix please link the issue you are fixing. Fixes #(issue)

如果是缺陷修复，请关联对应问题：Fixes #(问题编号)。

## Type of change

改动类型

Please delete options that are not relevant.

请删除不适用的选项。

- [ ] Bug fix (non-breaking change which fixes an issue)

  缺陷修复：修复问题，且不破坏既有行为。

- [ ] New Example (new example contribution)

  新增示例。

- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)

  不兼容改动：修复或功能会使现有功能无法按原先预期工作。

- [ ] This change requires a documentation update

  此改动需要更新文档。

## Feature/Issue validation/testing

功能或问题的验证与测试

Please describe the tests [UT/IT] that you ran to verify your changes and relevant result summary. If this is a bug fix please run `run_python_examples.sh` before and after your change locally to make sure it works and add the logs here.

请说明执行过的单元测试或集成测试及结果。如果是缺陷修复，请在修改前后分别在本地运行 run_python_examples.sh，并在此附上日志。

- [ ] Logs before change

  修改前的日志。

- [ ] Logs after change

  修改后的日志。

- Logs

  日志。

If this is a new example please add a corresponding test in `run_python_examples.sh`

如果新增示例，请在 run_python_examples.sh 中添加相应测试。

- [ ] Test Added

  已添加测试。

## Checklist:

检查清单

- [ ] Have you added tests that prove your fix is effective or that this example works?

  是否添加测试，证明修复有效或示例可以正常运行？

- [ ] Has code been commented, particularly in hard-to-understand areas?

  是否添加了代码注释，特别是难理解的部分？

- [ ] Have you made corresponding changes to the documentation?

  是否同步更新了相关文档？
