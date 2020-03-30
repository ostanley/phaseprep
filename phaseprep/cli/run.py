from argparse import ArgumentParser, RawTextHelpFormatter
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as ul
from phaseprep.workflows.preprocess_mag_wf import create_preprocess_mag_wf
from phaseprep.workflows.preprocess_phase_wf import create_preprocess_phase_wf
from phaseprep.workflows.feat_preproc_wf import create_feat_preproc_wf
import os
import os.path as op
from bids.layout import BIDSLayout
from nipype import config, logging
from phaseprep.interfaces import PhaseFitOdr
from nipype.utils.filemanip import split_filename


def get_tasklength(filename):
    # TODO: write block based specifity
    if isinstance(filelist, list):
        return [-1]*len(filelist)
    else:
        return -1


def get_restlength(filename):
    # TODO: write block based specifity
    if isinstance(filelist, list):
        return [-1]*len(filelist)
    else:
        return -1


def runpipeline(parser):
    # Parse inputs
    args = parser.parse_args()

    # 1) parse required inputs
    bids_dir = args.bids_dir
    participant = args.participant_label

    # 2) parse optional inputs
    nthreads = int(args.nthreads)
    bet_thr = float(args.bet_thr)
    small_fov = bool(args.small_fov)
    read_task_SNR = bool(args.taskSNR)

    # 2a) Need BIDS directory if no subjects chosen, use BIDSDataGrabber for this
    layout = BIDSLayout(bids_dir)

    if args.subjects:
        subjid = [s for s in args.subjects.split(',')]
        print(subjid)
    else:
        subjid = layout.get_subjects()

    # 2b) set directories for results
    deriv_dir = op.join(op.realpath(bids_dir), "derivatives")

    # Set work & crash directories
    if args.work_dir:
        work_dir = op.realpath(args.work_dir)
        crash_dir = op.join(op.realpath(args.work_dir), "crash")
    else:
        work_dir = op.join(bids_dir, "derivatives/work")
        crash_dir = op.join(op.join(op.realpath(bids_dir), "derivatives/work"), "crash")
    if len(subjid) == 1:
        work_dir = op.join(work_dir, subjid[0])
        crash_dir = op.join(work_dir, "crash")

    if not op.exists(work_dir):
        os.makedirs(work_dir)
    if not op.exists(crash_dir):
        os.makedirs(crash_dir)

    # 2c) set output directories
    if args.out_dir:
        out_dir = op.realpath(args.out_dir)
    else:
        out_dir = op.join(deriv_dir, 'phaseprep')

    config.update_config({'logging': {'log_directory': work_dir,
                                      'log_to_file': True,
                                      },
                          'execution': {'crashdump_dir': crash_dir,
                                        'crashfile_format': 'txt',
                                        'hash_method': 'content',
                                        'remove_unnecessary_outputs': False
                                        }})
    logging.update_logging(config)

    # BIDSDataGrabber
    layout = BIDSLayout(bids_dir, validate=False)

    if args.subjects:
        subject_id = [s for s in args.subjects.split(',')]
        print(type(subjid))
    else:
        subject_id = layout.get_subjects()

    # need to report on what subjects and what runs will be processed
    for subj in subjid:
        maglist = layout.get(subject=subject_id, datatype='func',
                             suffix='bold', extension=['nii', 'nii.gz'])
        phaselist = layout.get(subject=subject_id, datatype='func',
                               suffix='phase', extension=['nii', 'nii.gz'])
        print("Found ", max(len(maglist), len(phaselist)), " functional runs.")

        # get list of phase runs
        phaseruns = [f.run for f in phaselist]

        maglist_final = []
        phaselist_final = []
        # TODO: Match when there are multiple types of different runs
        for f in maglist:
            if f.run in phaseruns:
                pf = phaselist[phaseruns.index(f.run)]
                print("Processing magnitude run: ", f.run, "\n")

                # Check that runs have matching prefixes
                _, fname, _ = split_filename(f.filename)
                _, pfname, _ = split_filename(pf.filename)
                if fname[:-4] != pfname[:-5]:
                    continue

                # Check that runs have matching length, if not exclude.
                if 'dcmmeta_shape' in f.get_metadata().keys():
                    if f.get_metadata()['dcmmeta_shape'][-1] != pf.get_metadata()['dcmmeta_shape'][-1]:
                        continue

                # Check that runs have matching acq time, if not exclude.
                if 'AcquisitionTime' in f.get_metadata().keys():
                    if f.get_metadata()['AcquisitionTime'] != pf.get_metadata()['AcquisitionTime']:
                        continue

                maglist_final.append(f)
                phaselist_final.append(phaselist[phaseruns.index(f.run)])

        print(len(maglist_final), "runs with phase and magnitude were found.")
        print("Runs: ", [f.run for f in maglist_final])
        print("These runs have a matching length, name, and acquisition times.\n")

    # Step two will be magnitude preprocessing
    preproc_mag_wf = create_preprocess_mag_wf()
    preproc_mag_wf.inputs.inputspec.frac = bet_thr
    preproc_mag_wf.inputs.inputspec.input_mag = maglist_final
    preproc_mag_wf.inputs.extractor.robust = small_fov
    if read_task_SNR is True:
        preproc_mag_wf.inputs.inputspec.task = get_tasklength(maglist_final)
        preproc_mag_wf.inputs.inputspec.rest = get_restlength(maglist_final)
    else:
        preproc_mag_wf.inputs.inputspec.task = -1
        preproc_mag_wf.inputs.inputspec.rest = -1

    # Step three will be phase preprocessing
    preproc_phase_wf = create_preprocess_phase_wf()
    preproc_phase_wf.inputs.inputspec.input_mag = phaselist_final
    if read_task_SNR is True:
        preproc_mag_wf.inputs.inputspec.task = get_tasklength(phaselist_final)
        preproc_mag_wf.inputs.inputspec.rest = get_restlength(phaselist_final)
    else:
        preproc_phase_wf.inputs.inputspec.task = -1
        preproc_phase_wf.inputs.inputspec.rest = -1

    # Regress ge magnitude and phase
    # TODO: Function to get TR
    phaseregress = pe.MapNode(interface=PhaseFitOdr(), name='phaseregressodr', iterfield=['phase',
                                                                                          'mag'])
    phaseregress.inputs.noise_lb = 0.015

    # Step five will be ongoing during the previous steps ensuring correct sinking

    # Step six will be running this into a report

    phaseprep = pe.Workflow(name='phaseprep')
    phaseprep.base_dir = work_dir
    phaseprep.connect([(preproc_mag_wf, preproc_phase_wf, [('outputspec.motion_par',
                                                           'inputspec.motion_par'),
                                                           ('outputspec.mask_file',
                                                           'inputspec.mask_file')]),
                       (preproc_mag_wf, phaseregress, [('outputspec.proc_mag', 'mag')]),
                       ])

    print("setup pipline succesfully")
    if args.test is True:
        starttime = time()
        phaseprep.run(plugin='MultiProc', plugin_args={'n_procs': nthreads})
        print("completed pipeline in ", time()-starttime, " seconds.")



if __name__ == '__main__':
    from argparse import ArgumentParser, RawTextHelpFormatter

    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)

    # Required arguments
    g_req = parser.add_argument_group("Required arguments")
    g_req.add_argument("bids_dir", help="Directory with input dataset, "
                                        "formatted according to the BIDS "
                                        "standard")
    g_req.add_argument('participant_label', help="Participant id to perform "
                                                 "pipeline execution on (may be list)")

    # Optional arguments
    g_opt = parser.add_argument_group("Optional arguments")
    g_opt.add_argument("-s", "--subjects", dest="subjects",
                       help="List of subjects to limit analysis to delimit with ,")
    g_opt.add_argument("-b", "--bet_thr", dest="bet_thr", default=0.3,
                       help="User provided bet parameter, default 0.3")
    g_opt.add_argument("--small_fov", dest="small_fov", default=False,
                       help="In development: uses -Z for bet and may result in better images"
                       " if small FOV used")
    g_opt.add_argument("--taskbased_SNR", dest="taskSNR", default=False,
                       help="In development: reads block of task length from events.tsv in order to"
                       " calculate tSNR only during rest")
    g_opt.add_argument("-w", "--work_dir", dest="work_dir",
                       help="Work directory. Defaults to "
                       "<bids_dir>/derivatives/phaseprep/work")
    g_opt.add_argument("-o", "--out_dir", dest="out_dir",
                       help="Output directory. Defaults to "
                       "<bids_dir>/derivatives/phaseprep")
    g_opt.add_argument("-n", "--nthreads", dest="nthreads", default=2,
                       help="The number of threads to use "
                       "for pipeline execution where "
                       "applicable.")
    g_opt.add_argument("--fmriprep-dir", dest="fmriprep",
                       help="If fmriprep has been run and you wish to use"
                       " their transformations, the directory where the output"
                       " is stored")
    g_opt.add_argument("-t", "--test", dest="test", default=False,
                       help="Setup but do not run pipeline")
    runpipeline(parser)
