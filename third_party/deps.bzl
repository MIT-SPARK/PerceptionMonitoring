load("//third_party/boost:boost.bzl", "boost_repo")
load("//third_party/grante:grante.bzl", "grante_repo")
load("//third_party/gflags:gflags.bzl", "gflags_repo")
load("//third_party/gtest:gtest.bzl", "gtest_repo")
load("//third_party/glog:glog.bzl", "glog_repo")
load("//third_party/nlohmann_json:nlohmann_json.bzl", "nlohmann_json")
load("//third_party/csv:csv.bzl", "csv_repo")
load("//third_party/progressbar:progressbar.bzl", "progressbar_repo")
load("//third_party/opengm:opengm.bzl", "opengm_repo")

def deps_init():
    boost_repo()
    csv_repo()
    gflags_repo()
    glog_repo()
    grante_repo()
    gtest_repo()
    nlohmann_json()
    progressbar_repo()
    opengm_repo()
