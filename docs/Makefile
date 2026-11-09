# Minimal makefile for Sphinx documentation
# 用于构建 Sphinx 文档的最小 Makefile。
#

# You can set these variables from the command line, and also
# 可以通过命令行设置这些变量，前两个变量也可以
# from the environment for the first two.
# 通过环境变量设置。
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
# 将其放在首位，使不带参数的 make 等同于 make help。
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# 兜底目标：使用新的 make mode 选项，
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# 将所有未知目标交给 Sphinx；$(O) 是 $(SPHINXOPTS) 的简写。
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
