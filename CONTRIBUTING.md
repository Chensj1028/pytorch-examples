# Contributing to examples

参与 examples 项目贡献

We want to make contributing to this project as easy and transparent as
possible.

我们希望让参与本项目的过程尽可能简单、透明。

## Pull Requests

拉取请求（PR）

We actively welcome your pull requests.

欢迎提交拉取请求。

If you're new, we encourage you to take a look at issues tagged with [good first issue](https://github.com/pytorch/examples/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

初次参与时，建议查看标有 [good first issue](https://github.com/pytorch/examples/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) 的问题。

### For new examples

新增示例

0. Create a GitHub issue proposing a new example and make sure it's substantially different from an existing one.

   创建 GitHub 问题，提出新示例，并确保与现有示例有实质区别。

1. Fork the repo and create your branch from `main`.

   Fork 仓库，并从 main 创建分支。

2. If you've added code that should be tested, add tests to `run_python_examples.sh`.

   如果新增代码需要测试，请向 run_python_examples.sh 添加测试。

3. Create a `README.md`.

   创建 README.md。

4. Add a card with a brief description of your example and link to the repo to
   the `docs/source/index.rst` file and build the docs by running:

   在 docs/source/index.rst 中添加示例卡片，简要说明内容并链接到仓库，然后运行以下命令构建文档：

   ```
   cd docs
   virtualenv venv
   source venv/bin/activate
   pip install -r requirements.txt
   make html
   ```

   When done working with `virtualenv`, run `deactivate`.

   不再使用 virtualenv 时，运行 deactivate。

5. Verify that there are no issues in your doc build. You can check the preview locally
   by installing [sphinx-serve](https://pypi.org/project/sphinx-serve/)
   then running `sphinx-serve -b build`.

   确认文档构建无误。可安装 [sphinx-serve](https://pypi.org/project/sphinx-serve/)，再运行 sphinx-serve -b build，在本地预览。

6. Ensure your test passes locally.

   确保测试在本地通过。

7. If you haven't already, complete the Contributor License Agreement ("CLA").

   如果尚未签署，请完成贡献者许可协议（CLA）。

8. Address any feedback in code review promptly.

   及时处理代码评审中的反馈。

## For bug fixes

缺陷修复

1. Fork the repo and create your branch from `main`.

   Fork 仓库，并从 main 创建分支。

2. Make sure you have a GPU-enabled machine, either locally or in the cloud. `g4dn.4xlarge` is a good starting point on AWS.

   准备支持 GPU 的本地或云端机器。原指南推荐从 AWS 的 g4dn.4xlarge 开始。

3. Make your code change.

   修改代码。

4. Install `uv`.

   安装 uv。

5. Then, make sure that `VIRTUAL_ENV=.venv ./run_python_examples.sh` passes locally by running the script end to end.

   完整运行 VIRTUAL_ENV=.venv ./run_python_examples.sh，确保在本地通过。

6. If you haven't already, complete the Contributor License Agreement ("CLA").

   如果尚未签署，请完成贡献者许可协议（CLA）。

7. Address any feedback in code review promptly.

   及时处理代码评审中的反馈。

## Contributor License Agreement ("CLA")

贡献者许可协议（CLA）

To accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

项目接受 PR 前需要你签署 CLA。参与 Facebook 的任意开源项目，只需签署一次。

Complete your CLA here: <https://code.facebook.com/cla>

在此完成 CLA：<https://code.facebook.com/cla>。

## Issues

问题反馈

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

使用 GitHub Issues 跟踪公开缺陷。请清楚描述问题，并提供足够详细的复现步骤。

## License

许可证

By contributing to examples, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.

向 examples 贡献代码，即表示同意按本源码树根目录 LICENSE 文件中的条款许可你的贡献。
