import os
import streamlit as st
import altair as alt
from contextlib import contextmanager, redirect_stdout
from io import StringIO
import gurobipy as gp
import highspy
import pyscipopt
import mip
import pandas as pd
import time


def run_cbc(model: str, timelimit: int):

    m = mip.Model(solver_name=mip.CBC)

    m.read(model)
    start = time.time()
    status = m.optimize(max_seconds=timelimit)
    end = time.time()

    return {
        "solver": "CBC",
        "time": end-start,
        "iterations": None,
        "nodes": None,
        "gap": m.gap,
        "objective": m.objective_value,
        "status": status,
    }


def run_highs(model: str, timelimit: int):

    h = highspy.Highs()

    # Solve from mps file
    h.readModel(model)
    h.setOptionValue("time_limit", timelimit)
    if os.path.exists("highs.log"):
        os.remove("highs.log")
    h.setOptionValue("log_file", "highs.log")

    h.run()

    info = h.getInfo()
    version = []
    version.append(highspy.HIGHS_VERSION_MAJOR)
    version.append(highspy.HIGHS_VERSION_MINOR)
    version.append(highspy.HIGHS_VERSION_PATCH)

    return {
        "solver": f"HiGHS {version[0]}.{version[1]}.{version[2]}",
        "time": h.getRunTime(),
        "iterations": info.simplex_iteration_count,
        "nodes": info.mip_node_count,
        "gap": info.mip_gap,
        "objective": info.objective_function_value,
        "status": h.modelStatusToString(h.getModelStatus()),
    }


def run_pyscipopt(model: str, timelimit: int):

    m = pyscipopt.Model()
    m.redirectOutput()
    m.printVersion()
    m.readProblem(model)

    m.setParam("limits/time", timelimit)

    m.optimize()

    try:
        objval = m.getObjVal()
    except:
        objval = None

    return {
        "solver": f"SCIP {m.version()}",
        "time": m.getTotalTime(),
        "iterations": m.getNLPIterations(),
        "nodes": m.getNNodes(),
        "gap": m.getGap(),
        "objective": objval,
        "status": m.getStatus(),
    }


def run_gurobi(model: str, timelimit: int):

    m = gp.read(model)

    m.Params.TimeLimit = timelimit

    m.optimize()

    statuscodes = {
        1: "LOADED",
        2: "OPTIMAL",
        3: "INFEASIBLE",
        4: "INF_OR_UNBD",
        5: "UNBOUNDED",
        6: "CUTOFF",
        7: "ITERATION_LIMIT",
        8: "NODE_LIMIT",
        9: "TIME_LIMIT",
        10: "SOLUTION_LIMIT",
        11: "INTERRUPTED",
        12: "NUMERIC",
        13: "SUBOPTIMAL",
        14: "INPROGRESS",
        15: "USER_OBJ_LIMIT",
        16: "WORK_LIMIT",
        17: "MEM_LIMIT",
    }

    try:
        gap = m.MIPGap
    except:
        gap = None

    try:
        objval = m.ObjVal
    except:
        objval = None

    version = gp.gurobi.version()

    return {
        "solver": f"Gurobi {version[0]}.{version[1]}.{version[2]}",
        "iterations": m.IterCount,
        "nodes": m.NodeCount,
        "objective": objval,
        "gap": gap,
        "time": m.Runtime,
        "status": statuscodes[m.Status],
    }


@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret

        stdout.write = new_write
        yield


# parse arguments and pass to functions
if __name__ == "__main__":
    st.image(
        "https://raw.githubusercontent.com/Gurobi/.github/main/img/gurobi-light.png"
    )
    st.header("Benchmarking Gurobi and open-source solvers")
    model = st.file_uploader("Choose a model file", type=["mps", "lp"])

    with st.form("Run"):
        if model is not None:
            with open(model.name, "wb") as f:
                f.write(model.getvalue())

            st.write(f'File "{model.name}" has been uploaded successfully!')
            timelimit = st.slider("time limit in seconds", 1, 60)
            solvers = st.multiselect("Select solvers to compare:", ["Gurobi", "HiGHS", "SCIP", "CBC"], ["Gurobi", "HiGHS", "SCIP", "CBC"])
            submitted = st.form_submit_button("Run!")
            if submitted:
                results_list = []
                tabs = st.tabs(["Results"] + solvers)
                with st.spinner("Optimizing..."):
                    for i,s in enumerate(solvers):
                        if s == "Gurobi":
                            with tabs[i+1]:
                                output1 = st.empty()
                                with st_capture(output1.code):
                                    results_list.append(run_gurobi(model.name, timelimit))
                        if s == "HiGHS":
                            with tabs[i+1]:
                                results_list.append(run_highs(model.name, timelimit))
                                with open("highs.log", "r") as f:
                                    highslog = f.read()
                                st.code(highslog)
                        if s == "SCIP":
                            with tabs[i+1]:
                                output3 = st.empty()
                                with st_capture(output3.code):
                                    results_list.append(run_pyscipopt(model.name, timelimit))
                        if s == "CBC":
                            with tabs[i+1]:
                                output4 = st.empty()
                                with st_capture(output4.code):
                                    results_list.append(run_cbc(model.name, timelimit))
                with tabs[0]:
                    results = pd.DataFrame(results_list)
                    results.set_index("solver", inplace=True)
                    st.dataframe(results)
                    results.reset_index(inplace=True)
                    domain = results["solver"].tolist()
                    range = ["#DD2113", "green", "#1E3AC5", "#004746"]
                    chart = (
                        alt.Chart(results)
                        .mark_bar()
                        .encode(
                            x="time",
                            y="solver",
                            color=alt.Color(
                                "solver",
                                scale=alt.Scale(domain=domain, range=range),
                                title="solver",
                            ),
                        )
                    )
                    st.altair_chart(chart)
