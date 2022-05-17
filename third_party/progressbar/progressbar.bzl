load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def progressbar_repo():
    http_archive(
        name = "com_github_gipert_progressbar",
        build_file = "//third_party/progressbar:progressbar.BUILD",
        sha256 = "5dfe67d5ce3d6d16faa2ef7a80a66e826eea37d59a2c030557fb7f6f3e95d773",
        urls = ["https://github.com/gipert/progressbar/archive/refs/tags/v2.1.tar.gz"],
        strip_prefix = "progressbar-2.1/",
    )
