load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "factor_graph",
    hdrs = [
        "factor_graph.h",
    ],
    deps = [
        "//percival/diagnosability:dataset",
        "//percival/diagnosability:typedefs",
    ],
)

cc_library(
    name = "dummy",
    hdrs = ["dummy.h"],
    deps = [
        ":factor_graph",
        "//percival/diagnosability:diagnostic_graph",
    ],
)

cc_library(
    name = "grante",
    srcs = ["grante.cpp"],
    hdrs = ["grante.h"],
    deps = [
        ":factor_graph",
        ":grante_inference",
        "//percival/diagnosability:diagnostic_graph",
        "@boost//:bimap",
        "@com_github_nlohmann_json//:json",
        "@grante//grante",
    ],
)

cc_library(
    name = "grante_inference",
    srcs = ["grante_inference.cpp"],
    hdrs = ["grante_inference.h"],
    deps = [
        "@grante//grante",
    ],
)

cc_library(
    name = "opengm_typedefs",
    hdrs = ["opengm_typedefs.h"],
    deps = [
        ":factor_graph",
        "@com_github_opengm_opengm//:graphical_model",
    ],
)

cc_library(
    name = "opengm",
    srcs = ["opengm.cpp"],
    hdrs = [
        "opengm.h",
    ],
    deps = [
        ":opengm_inference",
        ":opengm_typedefs",
        "//percival/diagnosability:diagnostic_graph",
    ],
)

cc_library(
    name = "opengm_inference",
    srcs = ["opengm_inference.cpp"],
    hdrs = ["opengm_inference.h"],
    deps = [
        ":opengm_typedefs",
        "@com_github_opengm_opengm//:graphical_model",
        "@com_github_opengm_opengm//:inference",
    ],
)

cc_test(
    name = "grante_test",
    srcs = ["grante_test.cpp"],
    data = [
        "//percival/data:one_diagnosable.train.csv",
    ],
    deps = [
        ":grante",
        "//percival/diagnosability:example_graphs",
        "@com_google_googletest//:gtest_main",
    ],
)

# cc_library(
#     name = "gibbs",
#     hdrs = [
#         "gibbs.hpp",
#     ],
#     deps = [
#         "@com_github_opengm_opengm//:graphical_model",
#         "@com_github_opengm_opengm//:inference",
#         "@com_github_opengm_opengm//:operations",
#         "@com_github_opengm_opengm//:utilities",
#     ],
# )
