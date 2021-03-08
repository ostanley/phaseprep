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
from nipype.algorithms.confounds import TCompCor
import time
from niworkflows.interfaces.bids import DerivativesDataSink


def get_restlength(filename):
    # TODO: write block based specifity
    if isinstance(filename, list):
        print(filename)
        return [-1] * len(filename)
    else:
        return -1


def get_tasklength(filename):
    # TODO: write block based specifity
    if isinstance(filename, list):
        print(filename)
        return [-1] * len(filename)
    else:
        return -1


def get_TR(filename):
    import nibabel as nb

    if isinstance(filename, list):
        TR = []
        for f in filename:
            TR.append(nb.load(f).header.get_zooms()[-1])
    else:
        TR = nb.load(filename).header.get_zooms()[-1]
    return TR


def stripheader(filename):
    import numpy as np
    import os
    from nipype.utils.filemanip import split_filename

    if isinstance(filename, list):
        new_filename = []
        for f in filename:
            p, b, e = split_filename(f)
            new_f = b + "_noheader" + e
            data = np.loadtxt(f, skiprows=1)
            np.savetxt(new_f, data)
            new_filename.append(os.path.abspath(new_f))

    else:
        p, b, e = split_filename(filename)
        new_filename = os.path.abspath(b + "_noheader" + e)
        data = np.loadtxt(filename, skiprows=1)
        np.savetxt(new_filename, data)
    return new_filename


def get_magandphase(bids_dir, subject_id):
    from bids.layout import BIDSLayout
    from nipype.utils.filemanip import split_filename

    layout = BIDSLayout(bids_dir, validate=False)
    maglist = layout.get(
        subject=subject_id,
        datatype="func",
        suffix="bold",
        extension=[".nii", ".nii.gz"],
    )
    phaselist = layout.get(
        subject=subject_id,
        datatype="func",
        suffix="phase",
        extension=[".nii", ".nii.gz"],
    )
    print(f"Found {max(len(maglist), len(phaselist))} functional runs for {subject_id}.")

    # get list of phase runs
    phaseruns = [f.run for f in phaselist]

    maglist_final = []
    phaselist_final = []
    # TODO: Match when there are multiple types of different runs
    for f in maglist:
        if f.run in phaseruns:
            pf = phaselist[phaseruns.index(f.run)]

            # Check that runs have matching prefixes
            _, fname, _ = split_filename(f.filename)
            _, pfname, _ = split_filename(pf.filename)
            if fname[:-4] != pfname[:-5]:
                continue

            # Check that runs have matching length, if not exclude.
            if "dcmmeta_shape" in f.get_metadata().keys():
                if f.get_metadata()["dcmmeta_shape"][-1] != pf.get_metadata()["dcmmeta_shape"][-1]:
                    continue

            # Check that runs have matching acq time, if not exclude.
            if "AcquisitionTime" in f.get_metadata().keys():
                if f.get_metadata()["AcquisitionTime"] != pf.get_metadata()["AcquisitionTime"]:
                    continue

            maglist_final.append(f)
            phaselist_final.append(phaselist[phaseruns.index(f.run)])

    print(f"{len(maglist_final)} runs with phase and magnitude were found for {subject_id}.")
    print("Runs: ", [f.run for f in maglist_final])
    print("These runs have a matching length, name, and acquisition times.\n")

    return maglist_final, phaselist_final


def runpipeline(parser):
    # Parse inputs
    args = parser.parse_args()
    print(args)

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
        subject_id = [s for s in args.subjects.split(",")]
        print(subject_id)
    else:
        subject_id = layout.get_subjects()

    # 2b) set directories for results
    deriv_dir = op.join(op.realpath(bids_dir), "derivatives")

    # Set work & crash directories
    if args.work_dir:
        work_dir = op.realpath(args.work_dir)
        crash_dir = op.join(op.realpath(args.work_dir), "crash")
    else:
        work_dir = op.join(bids_dir, "derivatives/work")
        crash_dir = op.join(op.join(op.realpath(bids_dir), "derivatives/work"), "crash")
    if len(subject_id) == 1:
        work_dir = op.join(work_dir, subject_id[0])
        crash_dir = op.join(work_dir, "crash")

    if not op.exists(work_dir):
        os.makedirs(work_dir)
    if not op.exists(crash_dir):
        os.makedirs(crash_dir)

    # 2c) set output directories
    if args.out_dir:
        out_dir = op.realpath(args.out_dir)
    else:
        out_dir = op.join(deriv_dir, "phaseprep")

    config.update_config(
        {
            "logging": {
                "log_directory": work_dir,
                "log_to_file": True,
            },
            "execution": {
                "crashdump_dir": crash_dir,
                "crashfile_format": "txt",
                "hash_method": "content",
                "remove_unnecessary_outputs": False,
            },
        }
    )

    logging.update_logging(config)

    phaseprep = pe.Workflow(name="phaseprep")
    phaseprep.base_dir = work_dir
    sink_dict = {}

    infosource = pe.Node(interface=ul.IdentityInterface(fields=["subject_id"]), name="infosource")
    infosource.iterables = [("subject_id", subject_id)]

    filegrabber = pe.Node(
        ul.Function(
            function=get_magandphase,
            input_names=["bids_dir", "subject_id"],
            output_names=["maglist", "phaselist"],
        ),
        name="filegrabber",
    )
    filegrabber.inputs.bids_dir = bids_dir

    phaseprep.connect([(infosource, filegrabber, [("subject_id", "subject_id")])])

    # Step two will be magnitude preprocessing
    preproc_mag_wf = create_preprocess_mag_wf()
    preproc_mag_wf.inputs.inputspec.frac = bet_thr
    preproc_mag_wf.inputs.extractor.robust = small_fov

    sink_dict["procmag"] = pe.MapNode(
        DerivativesDataSink(
            desc="procmag",
            out_path_base="phaseprep",
            base_directory="out_dir",
            compress=True,
        ),
        name="dsink_procmag",
        iterfield=["in_file", "source_file"],
    )

    phaseprep.connect(
        [
            (
                filegrabber,
                preproc_mag_wf,
                [
                    ("maglist", "inputspec.input_mag"),
                    (("maglist", get_tasklength), "inputspec.task"),
                    (("maglist", get_restlength), "inputspec.rest"),
                ],
            ),
            (
                preproc_mag_wf,
                sink_dict["procmag"],
                [("outputspec.proc_mag", "in_file")],
            ),
            (filegrabber, sink_dict["procmag"], [("maglist", "source_file")]),
        ]
    )

    # Step three will be phase preprocessing
    preproc_phase_wf = create_preprocess_phase_wf()

    sink_dict["procphase"] = sink_dict["procmag"].clone("procphase")
    sink_dict["procphase"].inputs.desc = "procphase"

    phaseprep.connect(
        [
            (
                filegrabber,
                preproc_phase_wf,
                [
                    ("phaselist", "inputspec.input_phase"),
                    ("maglist", "inputspec.input_mag"),
                    (("phaselist", get_tasklength), "inputspec.task"),
                    (("phaselist", get_restlength), "inputspec.rest"),
                ],
            ),
            (
                preproc_mag_wf,
                preproc_phase_wf,
                [
                    ("outputspec.motion_par", "inputspec.motion_par"),
                    ("outputspec.mask_file", "inputspec.mask_file"),
                ],
            ),
            (
                preproc_phase_wf,
                sink_dict["procphase"],
                [("outputspec.proc_phase", "in_file")],
            ),
            (filegrabber, sink_dict["procphase"], [("phaselist", "source_file")]),
        ]
    )

    # Regress ge magnitude and phase
    phaseregress = pe.MapNode(
        interface=PhaseFitOdr.PhaseFitOdr(),
        name="phaseregressodr",
        iterfield=["phase", "mag", "TR"],
    )
    phaseregress.iterables = ("noise_lb", [0.1, 0.15, 0.25, 0.4])
    phaseregress.inputs.n_threads = 1

    sink_dict["micro"] = sink_dict["procmag"].clone("micro")
    sink_dict["micro"].inputs.desc = "micro"
    sink_dict["macro"] = sink_dict["procmag"].clone("macro")
    sink_dict["macro"].inputs.desc = "macro"
    sink_dict["r2"] = sink_dict["procmag"].clone("r2")
    sink_dict["r2"].inputs.desc = "r2"
    sink_dict["beta"] = sink_dict["procmag"].clone("beta")
    sink_dict["beta"].inputs.desc = "beta"

    phaseprep.connect(
        [
            (
                preproc_mag_wf,
                phaseregress,
                [
                    ("outputspec.proc_mag", "mag"),
                    (("outputspec.proc_mag", get_TR), "TR"),
                ],
            ),
            (preproc_phase_wf, phaseregress, [("outputspec.proc_phase", "phase")]),
            (phaseregress, sink_dict["macro"], [("filt", "in_file")]),
            (filegrabber, sink_dict["macro"], [("maglist", "source_file")]),
            (phaseregress, sink_dict["micro"], [("sim", "in_file")]),
            (filegrabber, sink_dict["micro"], [("maglist", "source_file")]),
            (phaseregress, sink_dict["r2"], [("corr", "in_file")]),
            (filegrabber, sink_dict["r2"], [("maglist", "source_file")]),
            (phaseregress, sink_dict["beta"], [("beta", "in_file")]),
            (filegrabber, sink_dict["beta"], [("maglist", "source_file")]),
        ]
    )

    # if tcompcor is true run with tcompcor applied as well
    if args.tcompcor:
        tcompcor = pe.MapNode(interface=TCompCor(), name="tcompcor", iterfield=["realigned_file"])

        stripheader_tcompcor = pe.MapNode(
            interface=ul.Function(
                function=stripheader,
                input_names=["filename"],
                output_names=["new_filename"],
            ),
            name="stripheader_tcompcor",
            iterfield=["filename"],
        )

        phaseregress_multi = pe.MapNode(
            interface=PhaseFitOdr.PhaseFitOdr(),
            name="phaseregress_multi",
            iterfield=["phase", "mag", "global_regressors", "TR"],
        )
        phaseregress_multi.inputs.n_threads = 1
        phaseregress_multi.iterables = ("noise_lb", [0.1, 0.15, 0.25, 0.4])

        sink_dict["micro_tcomp"] = sink_dict["procmag"].clone("micro_tcomp")
        sink_dict["micro_tcomp"].inputs.desc = "micro_tcomp"
        sink_dict["macro_tcomp"] = sink_dict["procmag"].clone("macro_tcomp")
        sink_dict["macro_tcomp"].inputs.desc = "macro_tcomp"
        sink_dict["r2_tcomp"] = sink_dict["procmag"].clone("r2_tcomp")
        sink_dict["r2_tcomp"].inputs.desc = "r2_tcomp"
        sink_dict["beta_tcomp"] = sink_dict["procmag"].clone("beta_tcomp")
        sink_dict["beta_tcomp"].inputs.desc = "beta_tcomp"

        phaseprep.connect(
            [
                (
                    preproc_mag_wf,
                    tcompcor,
                    [
                        ("outputspec.proc_mag", "realigned_file"),
                        ("outputspec.mask_file", "mask_files"),
                    ],
                ),
                (tcompcor, stripheader_tcompcor, [("components_file", "filename")]),
                (
                    stripheader_tcompcor,
                    phaseregress_multi,
                    [("new_filename", "global_regressors")],
                ),
                (
                    preproc_mag_wf,
                    phaseregress_multi,
                    [
                        ("outputspec.proc_mag", "mag"),
                        (("outputspec.proc_mag", get_TR), "TR"),
                    ],
                ),
                (
                    preproc_phase_wf,
                    phaseregress_multi,
                    [("outputspec.proc_phase", "phase")],
                ),
                (phaseregress_multi, sink_dict["macro_tcomp"], [("filt", "in_file")]),
                (filegrabber, sink_dict["macro_tcomp"], [("maglist", "source_file")]),
                (phaseregress_multi, sink_dict["micro_tcomp"], [("sim", "in_file")]),
                (filegrabber, sink_dict["micro_tcomp"], [("maglist", "source_file")]),
                (phaseregress_multi, sink_dict["r2_tcomp"], [("corr", "in_file")]),
                (filegrabber, sink_dict["r2_tcomp"], [("maglist", "source_file")]),
                (phaseregress, sink_dict["beta_tcomp"], [("beta", "in_file")]),
                (filegrabber, sink_dict["beta_tcomp"], [("maglist", "source_file")]),
            ]
        )
    # Step five will be ongoing during the previous steps ensuring correct sinking
    # Step six will be running this into a report

    print("setup pipline succesfully")
    if not args.test:
        print("running pipeline")
        starttime = time.time()
        phaseprep.write_graph(format="png")
        phaseprep.run(plugin="MultiProc", plugin_args={"n_procs": nthreads})
        print("completed pipeline in ", time.time() - starttime, " seconds.")


if __name__ == "__main__":
    from argparse import ArgumentParser, RawTextHelpFormatter

    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)

    # Required arguments
    g_req = parser.add_argument_group("Required arguments")
    g_req.add_argument(
        "bids_dir",
        help="Directory with input dataset, " "formatted according to the BIDS " "standard",
    )
    g_req.add_argument(
        "participant_label",
        help="Participant id to perform " "pipeline execution on (may be list)",
    )

    # Optional arguments
    g_opt = parser.add_argument_group("Optional arguments")
    g_opt.add_argument(
        "-s",
        "--subjects",
        dest="subjects",
        help="List of subjects to limit analysis to delimit with ,",
    )
    g_opt.add_argument(
        "-b",
        "--bet_thr",
        dest="bet_thr",
        default=0.3,
        help="User provided bet parameter, default 0.3",
    )
    g_opt.add_argument(
        "--small_fov",
        dest="small_fov",
        default=False,
        help="TODO In development: uses -Z for bet and may result in better images" " if small FOV used",
    )
    g_opt.add_argument(
        "--taskbased_SNR",
        dest="taskSNR",
        default=False,
        help="TODO In development: reads block of task length from events.tsv in order to"
        " calculate tSNR only during rest",
    )
    g_opt.add_argument(
        "-w",
        "--work_dir",
        dest="work_dir",
        help="Work directory. Defaults to " "<bids_dir>/derivatives/phaseprep/work",
    )
    g_opt.add_argument(
        "-o",
        "--out_dir",
        dest="out_dir",
        help="Output directory. Defaults to " "<bids_dir>/derivatives/phaseprep",
    )
    g_opt.add_argument(
        "-n",
        "--nthreads",
        dest="nthreads",
        default=2,
        help="The number of threads to use " "for pipeline execution where " "applicable.",
    )
    g_opt.add_argument(
        "--fmriprep-dir",
        dest="fmriprep",
        help="TODO If fmriprep has been run and you wish to use"
        " their transformations, the directory where the output"
        " is stored",
    )
    g_opt.add_argument(
        "-t",
        "--test",
        dest="test",
        action="store_true",
        help="Setup but do not run pipeline",
    )
    g_opt.add_argument(
        "--tcompcor",
        dest="tcompcor",
        action="store_true",
        help="also perform regression on tcompcor of data",
    )
    runpipeline(parser)
