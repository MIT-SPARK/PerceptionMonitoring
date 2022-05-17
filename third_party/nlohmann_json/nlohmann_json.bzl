load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def nlohmann_json():
    http_archive(
        name = "com_github_nlohmann_json",
        build_file = "//third_party/nlohmann_json:nlohmann_json.BUILD",
        sha256 = "b94997df68856753b72f0d7a3703b7d484d4745c567f3584ef97c96c25a5798e",
        urls = ["https://github.com/nlohmann/json/releases/download/v3.10.5/include.zip"],
    )
