"""Single entry point: python run_validation.py
Runs Passive + AARB/LQR1, verifies the result, and saves plots/CSV files.
No LaTeX, MATLAB, Simulink, PDF, or files outside this folder are needed.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import numpy as np
import scipy
import matplotlib
import parameters as p
from vehicle_model import STATE_NAMES
from simulation import simulate, rms, SIGNALS
from numerical_checks import verify
from plot_results import save_plots

BASE = Path(__file__).parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Optional output directory; relative paths are relative to the terminal directory.')
    parser.add_argument('--skip-checks',action='store_true',help='Skip Radau, step-halving and manuscript regression (for modified parameters).')
    args=parser.parse_args()
    output=args.output if args.output is not None else BASE/'outputs'
    output.mkdir(parents=True,exist_ok=True)
    if not np.isclose(p.ms+p.muf+p.mur,p.m):
        raise ValueError('Total mass m must equal ms + muf + mur. Update all relevant parameters consistently.')
    data=BASE/'data'
    steering=np.loadtxt(data/'steering.csv',delimiter=',',skiprows=1)
    reference={name:np.loadtxt(data/('reference_'+name+'.csv'),delimiter=',',skiprows=1) for name in SIGNALS}
    expected=json.loads((data/'expected_metrics.json').read_text())
    if expected['channel_order']!=SIGNALS:
        raise ValueError('Unexpected reference channel order.')
    print('1/4 Building vehicle models and designing LQR1 ...',flush=True)
    result=simulate(steering)
    t,y=result['t'],result['y']
    measured=rms(t,y)
    target=np.array(expected['printed_RMS_SI'])
    ref=np.column_stack([np.interp(t,reference[n][:,0],reference[n][:,1]) for n in SIGNALS])
    wave_error=100*rms(t,y-ref)/np.max(abs(ref),axis=0)
    print('2/4 Running numerical and manuscript regression checks ...',flush=True)
    checks=(dict(result['checks'],status='SKIPPED: independent solver and regression checks not run')
            if args.skip_checks else verify(result,steering,expected))
    csv_data=np.column_stack([t,y])
    np.savetxt(output/'time_histories.csv',csv_data,delimiter=',',header='time_s,'+','.join(SIGNALS),comments='')
    np.savetxt(output/'passive_time.csv',np.column_stack([t,y[:,:2]]),delimiter=',',header='time_s,Rf,Rr',comments='')
    np.savetxt(output/'active_time.csv',np.column_stack([t,y[:,2:]]),delimiter=',',header='time_s,Rf,Rr,uf_A,ur_A',comments='')
    np.savetxt(output/'active_states.csv',np.column_stack([t,result['xa']]),delimiter=',',header='time_s,'+','.join(STATE_NAMES),comments='')
    np.savetxt(output/'passive_states.csv',np.column_stack([t,result['xp']]),delimiter=',',header='time_s,'+','.join(STATE_NAMES[:6]),comments='')
    np.savetxt(output/'K_LQR1.csv',result['K'],delimiter=',')
    np.savez_compressed(output/'model_matrices.npz',A=result['active']['A'],B_control=result['active']['B'],
                        B_steering=result['active']['D'],C_LTR=result['active']['C'],K=result['K'],Q=result['Q'],R=result['R'],
                        A_passive=result['passive']['A'],B_steering_passive=result['passive']['D'],C_LTR_passive=result['passive']['C'])
    with (output/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['signal','unit','reference_printed_RMS','python_RMS','RMS_error_percent','waveform_error_percent'])
        for i,name in enumerate(SIGNALS):
            unit='mA' if i>=4 else 'dimensionless'; factor=1000 if i>=4 else 1
            writer.writerow([name,unit,target[i]*factor,measured[i]*factor,100*abs(measured[i]/target[i]-1),wave_error[i]])
    print('3/4 Drawing the four-panel comparison and separate time plots ...',flush=True)
    save_plots(result,reference,output)
    source_files=sorted(BASE.glob('*.py'))+sorted(data.glob('*'))
    report={
        'signal_order':SIGNALS,'RMS_SI':measured.tolist(),'printed_RMS_SI':target.tolist(),
        'RMS_error_percent':(100*abs(measured/target-1)).tolist(),
        'waveform_error_percent':wave_error.tolist(),
        'peak_reduction_percent':(100*(1-np.max(abs(y[:,2:4]),axis=0)/np.max(abs(y[:,:2]),axis=0))).tolist(),
        'numerical_checks':checks,
        'parameters':{name:value for name,value in vars(p).items() if not name.startswith('_') and isinstance(value,(int,float))},
        'geometry_provenance':'lact=0.50 m was selected following exploratory comparison with Figure 8; passive geometry is supplementary.',
        'reference_status':'Approximate digitized curves; numerical regression is not experimental validation or complete reference reproduction.',
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
        'source_sha256':{str(f.relative_to(BASE)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files if f.is_file()},
    }
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('4/4 Finished.\n')
    print('Signal       Python RMS       Printed RMS error')
    for i,name in enumerate(SIGNALS):
        factor=1000 if i>=4 else 1
        unit=' mA' if i>=4 else ''
        print(f'{name:12s} {measured[i]*factor:12.7f}{unit:3s}   {100*abs(measured[i]/target[i]-1):7.3f}%')
    print('\n'+checks['status'])
    print('Output folder:',output)
    print('Open Figure1_time_responses.png or time_Rf_Rr.png in that folder.')

if __name__=='__main__':
    main()