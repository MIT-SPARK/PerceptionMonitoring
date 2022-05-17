load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def grante_repo():
    git_repository(
        name = "grante",
        branch = "master",
        remote = "https://github.com/pantonante/grante-bazel.git"
    )

