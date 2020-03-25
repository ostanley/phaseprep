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


def findBIDSData(path):
    return path


def runpipeline(parser):
    # Parse inputs
    args = parser.parse_args()

    # 1) parse required inputs
    bids_dir = args.bids_dir
    participant = args.participant_label

    # 2) parse optional inputs
    nthreads = int(args.nthreads)
    bet_thr = float(args.bet_thr)
    radians = bool(args.radians)

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
        print(maglist, phaselist)
        print("Found ", max(len(maglist), len(phaselist)), " functional runs.")

        # get list of phase runs
        phaseruns = [f.run for f in phaselist]

        maglist_final = []
        phaselist_final = []
        for f in maglist:
            if f.run in phaseruns:
                # TODO: Check that runs have matching legth, if not exclude.
                pf = phaselist[phaseruns.index(f.run)]
                maglist_final.append(f)
                phaselist_final.append(phaselist[phaseruns.index(f.run)])

        print(len(maglist_final), " runs with phase and magnitude were found.\n")
        print("Runs: ", [f.run for f in maglist_final])
        # These runs have a matching length and phase regression will be preformed")

    # Step two will be magnitude preprocessing

    # Step three will be phase preprocessing

    # Step four will be running phase regression on the dataset

    # Step five will be ongoing during the previous steps ensuring correct sinking

    # Step six will be running this into a report

    print("ran pipline")


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
    g_opt.add_argument("--radians", dest="radians", default=False,
                       help="Data is in radians not siemens units")
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

    runpipeline(parser)
