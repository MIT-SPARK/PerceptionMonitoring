load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT license

cc_library(
    name = "base",
    hdrs = [
        "include/opengm/config.hxx",
        "include/opengm/datastructures/marray/marray.hxx",
        "include/opengm/opengm.hxx",
        "include/opengm/utilities/metaprogramming.hxx",
    ],
    strip_include_prefix = "include/",
)

cc_library(
    name = "utilities",
    hdrs = glob(["include/opengm/utilities/**"]),
    strip_include_prefix = "include/",
)

cc_library(
    name = "operations",
    hdrs = glob(["include/opengm/operations/**"]),
    strip_include_prefix = "include/",
)

cc_library(
    name = "functions",
    hdrs = glob(["include/opengm/functions/**"]),
    strip_include_prefix = "include/",
)

cc_library(
    name = "data_structures",
    hdrs = glob(["include/opengm/datastructures/**"]),
    strip_include_prefix = "include/",
)

cc_library(
    name = "inference",
    hdrs = glob(["include/opengm/inference/**"]),
    strip_include_prefix = "include/",
)

cc_library(
    name = "graphical_model",
    hdrs = glob(["include/opengm/graphicalmodel/**"]),
    strip_include_prefix = "include/",
    deps = [
        ":base",
        ":functions",
        ":operations",
        ":utilities",
        ":data_structures",
    ],
)
