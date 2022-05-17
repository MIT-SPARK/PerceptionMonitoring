load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def clean_dep(dep):
    return str(Label(dep))

def boost_repo():
    _RULES_BOOST_COMMIT = "652b21e35e4eeed5579e696da0facbe8dba52b1f"
    http_archive(
        name = "com_github_nelhage_rules_boost",
        sha256 = "c1b8b2adc3b4201683cf94dda7eef3fc0f4f4c0ea5caa3ed3feffe07e1fb5b15",
        strip_prefix = "rules_boost-%s" % _RULES_BOOST_COMMIT,
        # build_file = clean_dep("//third_party/boost:boost.BUILD"),
        urls = [
            "https://github.com/nelhage/rules_boost/archive/%s.tar.gz" % _RULES_BOOST_COMMIT,
        ],
    )
