"""Read-only provenance, explicit source adaptation, and original-R parity."""

from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pandas as pd

from .authors_models import AuthorsTrials, FREE, LIMITS, unpack

CLINICAL_AGE = ('b09h8s','M09R7M','o09b0u','b06g1z','m07s6z','b09u8r','s07s3m','s11r8m','k09z9z','l05d0p','k06v1t')
TASK_EXCLUSIONS = ('e06m9w','K05f9m','l02m7p','m04o8z','r02n9k','s04s8h','s10p5l')
REFERENCE = Path(__file__).resolve().parents[2]/'data/reference'


def phase_trials(source, participant, *, training):
    """Only requested phase/ID; validate labels before chronological sorting."""
    columns = ['ID','session','trial','stim1','stim2','action'] + (['reward1','reward2'] if training else [])
    frame = source.loc[source.ID.astype(str).eq(participant), columns].copy()
    if frame.empty or frame.isna().any().any():
        raise ValueError(f'{participant}: absent or missing required phase fields; no exclusions applied.')
    for col in ('session','trial','stim1','stim2'):
        numeric = pd.to_numeric(frame[col]).to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all() or (numeric<1).any():
            raise ValueError(f'Invalid integer labels in {col}.')
        frame[col] = numeric.astype(int)
    if frame.duplicated(['session','trial']).any() or not frame.action.isin(['left','right']).all():
        raise ValueError('Duplicate chronology or unknown actions; do not silently reinterpret.')
    ordered = frame.sort_values(['session','trial'], kind='stable')
    sorted_already = frame[['session','trial']].reset_index(drop=True).equals(ordered[['session','trial']].reset_index(drop=True))
    data = AuthorsTrials(ordered[['stim1','stim2']].to_numpy(int)-1,
                         ordered.action.eq('left').to_numpy(bool),
                         ordered[['reward1','reward2']].to_numpy(float) if training else None)
    return data, dict(rows=len(frame), original_order_chronological=sorted_already,
                      session_counts={str(k):int(v) for k,v in ordered.groupby('session').size().items()})


def participant_inputs(objects, participant):
    train, train_info = phase_trials(objects['data_4_analysis_RP_training'],participant,training=True)
    test, test_info = phase_trials(objects['data_4_analysis_RP_test'],participant,training=False)
    ratings = objects['ratings_RP2']
    rating_columns = [f's{i}_used_pre' for i in range(1,9)]
    if list(ratings.columns[1:16:2]) != rating_columns:
        raise ValueError('Driver positional rating columns differ from named stimulus order.')
    r = ratings.loc[ratings.ID.astype(str).eq(participant), rating_columns]
    if len(r)!=1:
        raise ValueError('Need exactly one participant rating row.')
    pre = objects['grouped_data_RP_pretest2_clean']
    p = pre.loc[pre.ID.astype(str).eq(participant), ['stim_chosen','perc_stim_chosen']].copy()
    p['stim_chosen'] = pd.to_numeric(p.stim_chosen).astype(int)
    if len(p)!=8 or set(p.stim_chosen)!=set(range(1,9)):
        raise ValueError('Need exactly eight supplied pretest choice proportions.')
    return train, test, r.iloc[0].to_numpy(float), p.sort_values('stim_chosen').perc_stim_chosen.to_numpy(float), dict(training=train_info,test=test_info)


def source_audit(objects, data_path):
    files = sorted(REFERENCE.glob('*.R')) + sorted(REFERENCE.glob('*.pdf'))
    assignments = []
    for path in REFERENCE.glob('*.R'):
        for number, line in enumerate(path.read_text().splitlines(),1):
            if re.search(r'\bpi\s*(?:<-|<<-|=)|assign\([\"\x27]pi',line):
                assignments.append(dict(file=path.name,line=number,text=line.strip()))
    ids = set(objects['data_4_analysis_RP_training'].ID.astype(str))
    audit = dict(objects={k:list(v.shape) if hasattr(v,'shape') else str(type(v)) for k,v in objects.items()},
                 workspace_pi_present='pi' in objects,
                 source_sha256=sha256(Path(data_path).read_bytes()).hexdigest(),
                 reference_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in files},
                 pi_assignments=assignments, exclusions=[dict(participant=p,category=c,present_in_training=p in ids)
                 for c,group in [('clinical_or_age',CLINICAL_AGE),('task_missing_accuracy_knowledge',TASK_EXCLUSIONS)] for p in group])
    if 'pi' in objects:
        value = np.asarray(objects['pi'],float)
        if value.size!=1 or not np.isfinite(value).all():
            raise ValueError('Workspace pi is not a finite scalar; inspect before literal scoring.')
        audit['literal_scoring_pi'] = float(value.item())
    else:
        audit['literal_scoring_pi'] = float(np.pi)
    if shutil.which('Rscript'):
        code = f'loaded<-load({json.dumps(str(Path(data_path).resolve()))}); cat("Loaded:",loaded,"\\n"); source({json.dumps(str(REFERENCE/"02_functions4modelling.R"))}); cat(R.version.string,"\\n"); cat(sprintf("%.17g",get("pi",environment(funky_train))),"\\n"); cat(sprintf("%.17g",get("pi",environment(funky_test))),"\\n"); cat(exists("pi",.GlobalEnv,inherits=FALSE))'
        audit['clean_r_namespace'] = subprocess.run(['Rscript','--vanilla','-e',code],check=True,capture_output=True,text=True).stdout
    return audit


def random_starts(participant_index, restarts, *, use_r=True):
    """Five draws in driver's order, indexed by PRETEST first-appearance order."""
    if restarts<1 or participant_index<1:
        raise ValueError('Positive restart count and one-based participant index required.')
    seed = 1234+17*participant_index
    if use_r:
        if not shutil.which('Rscript'):
            raise FileNotFoundError('Rscript needed for exact R starting draws; use explicit numpy RNG mode instead.')
        code = f'set.seed({seed}); x<-matrix(runif({5*restarts},min=c(.001,.001,.001,.001,.001),max=c(.999,9.999,.999,9.999,4.999)),ncol=5,byrow=TRUE); cat(sprintf("%.17g",as.vector(t(x))))'
        out = subprocess.run(['Rscript','--vanilla','-e',code],check=True,capture_output=True,text=True)
        return np.fromstring(out.stdout,sep=' ').reshape(restarts,5)
    return np.random.default_rng(seed).uniform([.001]*5,[.999,9.999,.999,9.999,4.999],(restarts,5))


def _r_vector(values):
    return 'c('+','.join(format(float(x),'.17g') for x in np.asarray(values).ravel())+')'


def original_r_evaluate(trials, model, theta, q0, h0, *, scoring_pi=None):
    """Unmodified helpers, fresh vanilla R. None deliberately leaves pi unbound.

    Test arrays have no rewards; placeholder zeros only satisfy unused R formals.
    Returned states/probabilities permit trial-level parity, not just total loss.
    """
    if not shutil.which('Rscript'):
        raise FileNotFoundError('Rscript unavailable for original-source parity.')
    ac,bc,ar,br,bias = unpack(model,theta)
    n = len(trials.left)
    rewards = trials.rewards if trials.rewards is not None else np.zeros((n,2))
    code = f'source({json.dumps(str(REFERENCE/"02_functions4modelling.R"))}); '
    code += f'n<-{n}; l<-{_r_vector(trials.pairs[:,0]+1)}; r<-{_r_vector(trials.pairs[:,1]+1)}; a<-{_r_vector(np.where(trials.left,1,2))}; '
    code += f'r1<-{_r_vector(rewards[:,0])}; r2<-{_r_vector(rewards[:,1])}; q<-matrix({_r_vector(q0)},1,8); h<-matrix({_r_vector(h0)},1,8); qt<-ht<-matrix(0,n+1,8); qt[1,]<-q; ht[1,]<-h; '
    if scoring_pi is not None:
        code += f'pi<-{float(scoring_pi):.17g}; '
    if trials.rewards is not None:
        code += f'objective<-funky_param_est({_r_vector(theta)},{model},l,r,r1,r2,a,h,ht,q,qt,n,rep(1,n)); '
        code += f'o<-funky_train({ac},{ac},{bc},{ar},{ar},{br},{bias},l,r,r1,r2,a,h,ht,q,qt,n,rep(1,n)); '
        code += 'qtrace<-o$value_rl_this_trial; '
    else:
        code += f'objective<-NA; o<-funky_test({ac},{ac},{bc},{ar},{ar},{br},{bias},l,r,r1,r2,a,h,ht,q,n); '
        code += 'qtrace<-matrix(rep(q,n+1),n+1,8,byrow=TRUE); '
    code += 'cat(sprintf("%.17g",c(objective,o$dev,o$choice_prob[,1],as.vector(t(qtrace)),as.vector(t(o$value_ck_this_trial))))); '
    out = subprocess.run(['Rscript','--vanilla','-e',code],check=True,capture_output=True,text=True)
    values = np.fromstring(out.stdout.replace('NA','nan'),sep=' ')
    if len(values)!=2+n+16*(n+1):
        raise ValueError('Unexpected original-R output schema.')
    return dict(objective=values[0],deviance=values[1],probability_left=values[2:2+n],
                q=values[2+n:2+n+8*(n+1)].reshape(n+1,8),h=values[2+n+8*(n+1):].reshape(n+1,8))


def original_r_fit(trials, model, q0, h0, starts, *, finite_difference_step=None):
    """Base-R optim on the original objective; not a claim of optimr parity."""
    n=len(trials.left); starts=np.asarray(starts,float); k=starts.shape[1]
    bounds=LIMITS[list(FREE[model])]
    code=f'source({json.dumps(str(REFERENCE/"02_functions4modelling.R"))}); '
    code+=f'n<-{n}; l<-{_r_vector(trials.pairs[:,0]+1)}; r<-{_r_vector(trials.pairs[:,1]+1)}; a<-{_r_vector(np.where(trials.left,1,2))}; '
    code+=f'r1<-{_r_vector(trials.rewards[:,0])}; r2<-{_r_vector(trials.rewards[:,1])}; q<-matrix({_r_vector(q0)},1,8); h<-matrix({_r_vector(h0)},1,8); qt<-ht<-matrix(0,n+1,8); qt[1,]<-q; ht[1,]<-h; '
    code+=f'starts<-matrix({_r_vector(starts)},ncol={k},byrow=TRUE); '
    code+=f'fn<-function(theta) funky_param_est(theta,{model},l,r,r1,r2,a,h,ht,q,qt,n,rep(1,n)); '
    extra='' if finite_difference_step is None else f',ndeps=rep({float(finite_difference_step)},{k})'
    code+=f'best<-NULL; for(i in 1:nrow(starts)) {{ o<-optim(starts[i,],fn,method="L-BFGS-B",lower={_r_vector(bounds[:,0])},upper={_r_vector(bounds[:,1])},control=list(factr=1e4,pgtol=1e-7,maxit=1000{extra})); if(is.null(best)||o$value<best$value) best<-o; }}; '
    code+='cat(sprintf("%.17g",c(best$value,best$convergence,best$par))); '
    out=subprocess.run(['Rscript','--vanilla','-e',code],check=True,capture_output=True,text=True)
    values=np.fromstring(out.stdout,sep=' ')
    return dict(deviance=float(values[0]),convergence=int(values[1]),theta=values[2:].tolist())
