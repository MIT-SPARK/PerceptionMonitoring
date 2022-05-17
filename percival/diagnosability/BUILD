load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "typedefs",
    hdrs = ["typedefs.h"],
)

cc_library(
    name = "dataset",
    srcs = ["dataset.cpp"],
    hdrs = ["dataset.h"],
    deps = [
        ":typedefs",
        "@com_github_vincentlaucsb_csv//:csv",
    ],
)

cc_library(
    name = "weights",
    srcs = [
        "weights.cpp",
    ],
    hdrs = [
        "weights.h",
    ],
    deps = [":typedefs"],
)

cc_library(
    name = "diagnostic_graph",
    srcs = [
        "utils.cpp",
    ],
    hdrs = [
        "diagnostic_graph.h",
        "diagnostic_graph.tpp",
        "diagnostic_graph_typedefs.h",
        "utils.h",
    ],
    deps = [
        ":weights",
        ":dataset",
        "//percival/diagnosability/factor_graph:factor_graph",
        "@com_github_google_glog//:glog",
        "@com_github_nlohmann_json//:json",
    ],
)

cc_library(
    name = "example_graphs",
    hdrs = ["example_graphs.hpp"],
    deps = [
        ":diagnostic_graph",
    ],
)

cc_library(
    name = "powerset",
    hdrs = ["powerset.hpp"],
)

cc_test(
    name = "weights_test",
    srcs = ["weights_test.cpp"],
    deps = [
        ":weights",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "diagnostic_graph_test",
    srcs = ["diagnostic_graph_test.cpp"],
    data = [
        "//percival/data:dummy.json",
        "//percival/data:one_diagnosable.train.csv",
    ],
    deps = [
        ":dataset",
        ":diagnostic_graph",
        ":example_graphs",
        "//percival/diagnosability/factor_graph:dummy",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cpp"],
    deps = [
        ":diagnostic_graph",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "dataset_test",
    srcs = ["dataset_test.cpp"],
    data = [
        "//percival/data:one_diagnosable.train.csv",
    ],
    deps = [
        ":dataset",
        "@com_google_googletest//:gtest_main",
    ],
)
