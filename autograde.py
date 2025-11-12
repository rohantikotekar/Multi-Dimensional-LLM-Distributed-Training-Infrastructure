#!/usr/bin/env python

import click
import os
import json
import re
#from csvtools import qcsv
#from CSE142L.jextract import extract as qjson
#from notebook import render_csv
import pandas as pd

def compute_speedup(dir=None, tests=None):
    speedup = 0
    if dir == None:
        dir=""
    if tests == None:
        tests = 10
    def csv(f):
        return pd.read_csv(f, sep=",")

    corrects = 0

    baseline = csv(os.path.join(dir, "baseline.csv"))
    results = csv(os.path.join(dir, "my_train.csv"))
    speedup = float(baseline.iloc[0][" ET"])/float(results.iloc[0][" ET"])
#    print(baseline.iloc[0])
#    print(results.iloc[0])
#    print(speedup)
    return speedup

def compute_correctness(dir=None, tests=None):
    if dir == None:
        dir=""
    if tests == None:
        tests = 10
    def csv(f):
        return pd.read_csv(f, sep=",")

    corrects = 0

    baseline = csv(os.path.join(dir, "baseline.csv"))
    results = csv(os.path.join(dir, "my_train.csv"))
    for i in range(tests):
        if abs(baseline.iloc[0]["Loss: "+str(i)] - results.iloc[0]["Loss: "+str(i)])<=0.000002:
            corrects = corrects+1
    return corrects

@click.command()
@click.option("--submission", required=True,  type=click.Path(exists=True), help="Test directory")
@click.option("--results", required=True, type = click.File(mode="w"), help="Where to put results")
def autograde(submission=None, results=None):
    correct = 0;
    speedup = 0;
    try:
        correctness = compute_correctness(dir=submission)
        failures = 10 - correctness
        output = "tests passed"
        if failures == 0:
            correct = 1
        else:
            output = f"Your code is incorrect"
    except FileNotFoundError as e:
        output = f"I couldn't find your regression outputs.  This often means your program generated a segfault :{e}."
        failures = 1
    except Exception as e:
        output = f"I got an unexpected exception processing the regressions.  Tell the course staff:{e}."
        failures = 1
    finally:
        regressions = dict(score=1 if failures == 0 else 0,
                           max_score=1,
                           number="1",
                           output=output,
                           tags=[],
                           visibility="visible")
    try:
        speedup  = compute_speedup(dir=submission)
    except FileNotFoundError as e:
        output = f"I couldn't find a csv file.  This often means your program generated a segfault or failed the regressions :{e}."
    except Exception as e:
        output = f"I got an unexpected exception evaluating the benchmarks.  Tell the course staff.:{e}."

    if os.path.exists("/autograder/results/stdout"):
        with open("/autograder/results/stdout") as f:
            stdout = f.read()
    else:
        stdout = ""
    
    score = correctness*(speedup-1)*10000/69
    if(score > 100):
        score = 100
    elif(score < 0):
        score = 0

    # https://gradescope-autograders.readthedocs.io/en/latest/specs/#output-format
    json.dump(dict(output=stdout,
                   visibility="visible",
                   stdout_visibility="visible",
                   tests=[ dict(score=score,
                                max_score=100,
                                number="1",
                                correctness=correctness,
                                speedup=speedup,
                                output=stdout,
                                tags=[],
                                visibility="visible",)
                   ],
                   leaderboard=speedup,),
                   results, sort_keys=True, indent=4)
      
if __name__== "__main__":
#    compute_speedup()
    autograde()
