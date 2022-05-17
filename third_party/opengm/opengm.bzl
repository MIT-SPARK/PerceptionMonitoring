# load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def opengm_repo():
    new_git_repository(
        name = "com_github_opengm_opengm",
        commit = "decdacf4caad223b0ab5478d38a855f8767a394f",
        remote = "https://github.com/opengm/opengm.git",
        build_file= "//third_party/opengm:opengm.BUILD", 
    )