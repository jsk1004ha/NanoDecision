"""Probability post-processing. Standard library only in the runtime."""
import math


def softmax(logits, temperature=1.0):
    if not logits or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid logits or temperature")
    if not all(math.isfinite(float(x)) for x in logits):
        raise ValueError("Non-finite logits")
    maximum = max(logits)
    values = [math.exp((x-maximum)/temperature) for x in logits]
    den = sum(values)
    return [x/den for x in values]


def check_distribution(p):
    if len(p) < 2 or not all(math.isfinite(x) and 0 <= x <= 1 for x in p) or abs(sum(p)-1)>1e-5:
        raise ValueError("Expected a finite probability distribution summing to one")


def metrics(probabilities, target, values=None):
    check_distribution(probabilities); check_distribution(target)
    if len(probabilities) != len(target):
        raise ValueError("Target length mismatch")
    nll = -sum(y*math.log(max(p, 1e-30)) for p,y in zip(probabilities,target))
    # Brier to a soft target omits the target-only variance constant. For hard
    # labels this is the conventional sum-of-squares multiclass Brier score.
    brier = sum((p-y)**2 for p,y in zip(probabilities,target))
    result = {"nll": nll, "brier_to_target": brier}
    if values is not None:
        if len(values)!=len(target) or any(b<=a for a,b in zip(values,values[1:])):
            raise ValueError("Score values must be strictly increasing")
        cp=ct=rps=0.0
        span=values[-1]-values[0]
        for i in range(len(values)-1):
            cp+=probabilities[i];ct+=target[i]
            rps+=(cp-ct)**2*(values[i+1]-values[i])/span
        result["normalized_crps"] = rps
    hard = max(target) > 1-1e-7
    if hard:
        result["correct"] = int(max(range(len(target)),key=probabilities.__getitem__)==target.index(max(target)))
    return result


def fit_temperature(records):
    """Fit one scalar on a dedicated temperature split; model must be fixed."""
    rows = list(records)
    if not rows:
        raise ValueError("Empty temperature split")
    def objective(log_t):
        t = math.exp(log_t); total=0.0
        for r in rows:
            z=r["logits"];q=r["target_probs"];check_distribution(q)
            if len(z)!=len(q):raise ValueError("Logit/label mismatch")
            m=max(z);lse=math.log(sum(math.exp((a-m)/t) for a in z))
            total += sum(y*(lse-(a-m)/t) for a,y in zip(z,q))
        return total/len(rows)
    # A bounded one-dimensional optimizer. T=1 is an explicit fallback.
    a,b=math.log(0.05),math.log(20.0);golden=(math.sqrt(5)-1)/2
    c=b-golden*(b-a);d=a+golden*(b-a);fc,fd=objective(c),objective(d)
    for _ in range(70):
        if fc<fd:b,d,fd=d,c,fc;c=b-golden*(b-a);fc=objective(c)
        else:a,c,fc=c,d,fd;d=a+golden*(b-a);fd=objective(d)
    candidate=(a+b)/2
    if objective(candidate)>objective(0.0):candidate=0.0
    return {"temperature": math.exp(candidate),"before_nll": objective(0.0),
            "after_nll": objective(candidate),"n":len(rows)}


def conformal_quantile(scores, alpha):
    if not 0<alpha<1 or not scores or not all(math.isfinite(s) and 0<=s<=1 for s in scores):
        raise ValueError("Invalid conformal scores/alpha")
    n=len(scores);k=math.ceil((n+1)*(1-alpha))
    # LAC scores are <=1. k=n+1 corresponds to an all-label prediction set.
    return 1.0 if k>n else sorted(scores)[k-1]


def prediction_set(probabilities, qhat):
    check_distribution(probabilities)
    if not math.isfinite(qhat) or not 0<=qhat<=1:raise ValueError("Invalid conformal quantile")
    return [i for i,p in enumerate(probabilities) if 1-p <= qhat+1e-12]


def bayes_action(probabilities, cost_matrix, abstain_cost=None):
    """cost_matrix[action][outcome]. Minimize expected loss; no guarantee of p."""
    check_distribution(probabilities)
    if not cost_matrix or any(len(row)!=len(probabilities) for row in cost_matrix):
        raise ValueError("Invalid cost matrix")
    if not all(math.isfinite(v) for row in cost_matrix for v in row):
        raise ValueError("Non-finite cost")
    risk=[sum(c*p for c,p in zip(row,probabilities)) for row in cost_matrix]
    index=min(range(len(risk)),key=risk.__getitem__)
    if abstain_cost is not None:
        if not math.isfinite(abstain_cost):raise ValueError("Non-finite abstention cost")
        if abstain_cost<=risk[index]:return {"action":None,"expected_cost":abstain_cost,"risks":risk}
    return {"action":index,"expected_cost":risk[index],"risks":risk}


class EvaluationAccumulator:
    """Streaming diagnostics; fixed bins/thresholds are not fitted on test data."""
    def __init__(self, qhat=None, bins=15):
        if bins < 2: raise ValueError("At least two bins required")
        self.qhat, self.bins = qhat, [[0, 0.0, 0] for _ in range(bins)]
        self.n = self.hard = self.correct = self.ordinal_n = 0
        self.nll = self.brier = self.rps = self.score_mae = 0.0
        self.groups = {}; self.covered = self.set_size = self.singletons = self.singleton_correct = 0
        self.thresholds = {x: [0, 0] for x in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)}

    def add(self, p, q, values=None, group_id=None):
        m = metrics(p, q, values)
        self.n += 1; self.nll += m["nll"]; self.brier += m["brier_to_target"]
        if "normalized_crps" in m:
            self.ordinal_n += 1; self.rps += m["normalized_crps"]
            self.score_mae += abs(sum(a*v for a,v in zip(p,values))-sum(a*v for a,v in zip(q,values)))
        if "correct" not in m: return
        self.hard += 1; self.correct += m["correct"]
        confidence = max(p); index = min(int(confidence*len(self.bins)), len(self.bins)-1)
        cell = self.bins[index]; cell[0] += 1; cell[1] += confidence; cell[2] += m["correct"]
        for threshold, cell in self.thresholds.items():
            if confidence >= threshold: cell[0] += 1; cell[1] += m["correct"]
        if self.qhat is not None:
            labels = prediction_set(p, self.qhat); truth = q.index(max(q)); covered = truth in labels
            self.covered += covered; self.set_size += len(labels)
            if len(labels)==1:
                self.singletons += 1; self.singleton_correct += labels[0]==truth
            if group_id is not None:
                self.groups[group_id] = self.groups.get(group_id, True) and covered

    def summary(self):
        if not self.n: raise ValueError("Empty evaluation")
        result = {"count":self.n, "hard_label_count":self.hard,
                  "nll":self.nll/self.n, "brier_to_target":self.brier/self.n,
                  "accuracy":self.correct/self.hard if self.hard else None,
                  "ece_fixed_bins":sum(abs(s-c) for n,s,c in self.bins)/self.hard if self.hard else None,
                  "ece_bin_count":len(self.bins),
                  "ordinal_normalized_crps":self.rps/self.ordinal_n if self.ordinal_n else None,
                  "score_expectation_mae_to_target_mean":self.score_mae/self.ordinal_n if self.ordinal_n else None,
                  "fixed_threshold_diagnostics":{str(t):{"coverage":n/self.hard if self.hard else None,
                      "accepted_accuracy":c/n if n else None,"accepted_count":n} for t,(n,c) in self.thresholds.items()}}
        if self.qhat is not None:
            result["conformal"]={"row_coverage":self.covered/self.hard if self.hard else None,
                "group_simultaneous_coverage":sum(self.groups.values())/len(self.groups) if self.groups else None,
                "independent_group_count":len(self.groups),"average_set_size":self.set_size/self.hard if self.hard else None,
                "singleton_fraction":self.singletons/self.hard if self.hard else None,
                "singleton_accuracy":self.singleton_correct/self.singletons if self.singletons else None,
                "note":"Observed diagnostics, not a conditional or accepted-only guarantee."}
        return result
