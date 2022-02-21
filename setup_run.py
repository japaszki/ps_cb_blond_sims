import os
import pickle

def setup_run(run_dir, source_dir, params, job_flavour):
    #Create root directory for this run:
    try:
        os.makedirs(run_dir)
    except:
        pass   
    
    #Create directories for Condor outputs:
    try:
        os.makedirs(run_dir + 'log/')
    except:
        pass
    
    try:
        os.makedirs(run_dir + 'output/')
    except:
        pass
    
    try:
        os.makedirs(run_dir + 'error/')
    except:
        pass
    
    try:
        os.makedirs(run_dir + 'sim_outputs/')
    except:
        pass
    
    try:
        os.makedirs(run_dir + 'sim_outputs/cb_plots/')
    except:
        pass
    
    try:
        os.makedirs(run_dir + 'sim_outputs/profile_plots/')
    except:
        pass
    
    #Copy bash script:
    os.system('cp ' + source_dir + 'run.sh ' + run_dir)
    
    #Copy submit file, setting appropriate jobflavour:
    os.system('cp ' + source_dir + 'run.sub ' + run_dir)
    
    submit_file_base = open(source_dir + 'run.sub','r')
    submit_file_write = open(run_dir +'run.sub','w')

    for line in submit_file_base:
        line = line.replace('JOBFLAVOUR', job_flavour)
        submit_file_write.write(line)
    
    submit_file_base.close()
    submit_file_write.close()
    
    #Create pickle file with parameters:
    with open(run_dir + '/input_params.pickle', 'wb') as f:
        pickle.dump(params, f)

    #Submit job:
    os.chdir(run_dir)
    print('Submitting to Condor from: ' + os.getcwd())
    os.system('condor_submit run.sub')