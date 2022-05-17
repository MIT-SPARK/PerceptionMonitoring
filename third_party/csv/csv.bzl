load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def csv_repo():
    http_archive(
        name = "com_github_vincentlaucsb_csv",
        build_file = "//third_party/csv:csv.BUILD",
        sha256 = "3f6ce9212e66d273de12a9671dcbf7be7da0241334dc690585dd434dce5e5acf",
        urls = ["https://github.com/vincentlaucsb/csv-parser/archive/refs/tags/2.1.3.tar.gz"],
        strip_prefix = "csv-parser-2.1.3/",
    )
