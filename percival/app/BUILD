load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "example",
    srcs = ["example.cpp"],
    deps = [
        "//percival/diagnosability:diagnostic_graph",
        "//percival/diagnosability:example_graphs",
        "//percival/diagnosability/factor_graph:grante",
        "//percival/diagnosability/factor_graph:opengm",
    ],
)

cc_binary(
    name = "from_file",
    srcs = ["from_file.cpp"],
    deps = [
        "//percival/diagnosability:diagnostic_graph",
        "//percival/diagnosability/factor_graph:grante",
        "@com_github_gflags_gflags//:gflags",
    ],
)

# cc_binary(
#     name = "nextb",
#     srcs = ["nextb.cpp"],
#     deps = [
#         "//percival/diagnosability:powerset",
#     ],
# )

cc_binary(
    name = "prova",
    srcs = ["prova.cpp"],
    deps = [
        "//percival/diagnosability:dataset",
        "//percival/diagnosability:diagnostic_graph",
        "//percival/diagnosability/factor_graph:grante",
        "@com_github_gflags_gflags//:gflags",
    ],
)

cc_binary(
    name = "train",
    srcs = ["train.cpp"],
    deps = [
        "//percival/diagnosability:dataset",
        "//percival/diagnosability:diagnostic_graph",
        "//percival/diagnosability/factor_graph:grante",
        "@com_github_gflags_gflags//:gflags",
        "@com_github_gipert_progressbar//:progressbar",
    ],
)

cc_binary(
    name = "inference",
    srcs = ["inference.cpp"],
    deps = [
        "//percival/diagnosability:dataset",
        "//percival/diagnosability:diagnostic_graph",
        "//percival/diagnosability/factor_graph:grante",
        "@com_github_gflags_gflags//:gflags",
        "@com_github_gipert_progressbar//:progressbar",
    ],
)
