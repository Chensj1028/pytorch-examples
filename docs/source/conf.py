#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# 版权所有 © Meta Platforms, Inc. 及其关联方。
# All rights reserved.
# 保留所有权利。
#
# This source code is licensed under the BSD-style license found in the
# 此源码使用 BSD 风格许可证，详见
# LICENSE file in the root directory of this source tree.
# 源码树根目录中的 LICENSE 文件。

# Configuration file for the Sphinx documentation builder.
# Sphinx 文档构建器的配置文件。
#
# This file only contains a selection of the most common options. For a full
# 此文件仅包含最常用选项中的一部分；完整
# list see the documentation:
# 列表见文档：
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
# 路径设置。

# If extensions (or modules to document with autodoc) are in another directory,
# 如果扩展或需要 autodoc 处理的模块位于其他目录，
# add these directories to sys.path here. If the directory is relative to the
# 在这里将这些目录加入 sys.path。如果路径相对于
# documentation root, use os.path.abspath to make it absolute, like shown here.
# 文档根目录，应按示例用 os.path.abspath 转为绝对路径。
#
import os
import sys

import pytorch_sphinx_theme

current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, target_dir)
print(target_dir)

# -- Project information -----------------------------------------------------
# 项目信息。

project = "PyTorchExamples"
copyright = "2022, Meta"
author = "Meta"

# The full version, including alpha/beta/rc tags
# 完整版本号，包括 alpha、beta、rc 等标签。
release = "1.11"

# -- General configuration ---------------------------------------------------
# 常规配置。

# Add any Sphinx extension module names here, as strings. They can be
# 在这里以字符串形式添加 Sphinx 扩展模块名称；可以使用
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# Sphinx 内置扩展（名称为 sphinx.ext.*），也可以使用
# ones.
# 自定义扩展。
extensions = ["sphinx.ext.napoleon", "sphinx.ext.autodoc", "sphinx_panels"]

# Add any paths that contain templates here, relative to this directory.
# 在此添加模板目录路径，路径相对于当前目录。
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# 用于匹配忽略文件和目录的模式列表，
# directories to ignore when looking for source files.
# 查找源文件时会跳过这些项目；路径相对于源码目录。
# This pattern also affects html_static_path and html_extra_path.
# 此模式也会影响 html_static_path 和 html_extra_path。
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# HTML 输出选项。

# The theme to use for HTML and HTML Help pages.  See the documentation for
# HTML 和 HTML Help 页面的主题；内置主题列表
# a list of builtin themes.
# 可参阅文档。
#
# html_theme = 'alabaster'
html_theme = "pytorch_sphinx_theme"
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# 在此添加自定义静态文件目录，例如样式表目录；路径
# relative to this directory. They are copied after the builtin static files,
# 相对于当前目录。这些文件在内置静态文件之后复制，
# so a file named "default.css" will overwrite the builtin "default.css".
# 因此名为 default.css 的文件会覆盖内置的同名文件。

html_static_path = ["_static"]
panels_add_fontawesome_latex = True

html_theme_options = {
    'pytorch_project': 'examples',
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': False,
    'analytics_id': 'UA-117752657-2',
}
