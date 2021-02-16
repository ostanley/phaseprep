from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec
from nipype.utils.filemanip import split_filename
from scipy import odr as odr
import nibabel as nb
import numpy as np
import os


class PhaseFitOdr_RefactorInputSpec(BaseInterfaceInputSpec):
    phase = File(exists=True, desc='phase image', mandatory=True)
    mag = File(exists=True, desc='mag image', mandatory=True)
    TR = traits.Float(desc='repetition time of the scan', mandatory=True)
    noise_lb = traits.Float(desc='Noise filter lower bound; should be higher'
                                 ' than pyiological signal')
    global_regressors = File(exists=True, desc='File of regressors (.tsv)')
    n_threads = traits.Int(desc="number of threads to limit pool to,"
                                "fixes numpy api to use all threads")


class PhaseFitOdr_RefactorOutputSpec(TraitedSpec):
    sim = File(exists=True, desc="sim")
    filt = File(exists=True, desc="filt")
    corr = File(exists=True, desc="corr")
    corradj = File(exists=True, desc="corradj")
    residuals = File(exists=True, desc="residuals")
    stdm = File(exists=True, desc="std error in mag")
    stdp = File(exists=True, desc="std error in phase")
    exit = File(exists=True, desc="int code for stopping")
    beta = File(exists=True, desc="betas from fit")


class PhaseFitOdr_Refactor(BaseInterface):
    """Nipype Interface to fit phase and magnitude data together

    Inputs:

    Outputs:

    """
    input_spec = PhaseFitOdr_RefactorInputSpec
    output_spec = PhaseFitOdr_RefactorOutputSpec

    def multiplelinear(self, beta, x):
        f = np.zeros(x[0].shape)
        for r in range(x.shape[0]):
            f += beta[r]*x[r]
        return f

    def get_noise_est(self, ts):
        noise = np.fft.ifft((np.abs(np.fft.fftfreq(ts.shape[-1], self.inputs.TR)) > self.inputs.noise_lb) *
                            np.fft.fft(ts - np.mean(ts)))
        return np.std(noise), noise

    def run_ODR(self, mag, phase, regressors=None):
        linearfit = odr.Model(self.multiplelinear)

        stdm, noise = self.get_noise_est(mag)
        stdp, _ = self.get_noise_est(phase)

        if regressors is not None:
            design = np.concatenate((phase[:, np.newaxis], regressors,
                                     np.ones_like(phase[:, np.newaxis])), axis=1).T
            ests = np.concatenate(([stdm/stdp],
                                   np.std(regressors, axis=0)/stdp,
                                   [np.mean(mag) / np.mean(phase)]))
            mydata = odr.RealData(design, mag,
                                  sx=np.concatenate(([stdp],
                                                     np.std(
                                                         regressors, axis=0),
                                                     [1])),
                                  sy=stdm)
        else:
            design = np.row_stack((phase, np.ones(phase.shape)))
            ests = [np.std(mag) / np.std(phase),
                    np.mean(mag) / np.mean(phase)]
            mydata = odr.RealData(design, mag,
                                  sx=np.hstack([stdp, np.finfo(stdp).eps]),
                                  sy=stdm)

        # and fit model
        # mag = A*phase + B*regressors + C
        # (y=mx+b)
        # call : (x,y,sx,sy)
        odr_obj = odr.ODR(mydata, linearfit, beta0=ests, maxit=400)
        res = odr_obj.run()
        return res, design

    def _run_interface(self, runtime):
        f = nb.load(self.inputs.mag)
        mag = f.get_data()

        f = nb.load(self.inputs.phase)
        ph = f.get_data()

        saveshape = np.array(mag.shape)
        nt = mag.shape[-1]

        if self.inputs.global_regressors:
            regressors = np.loadtxt(self.inputs.global_regressors)
            if len(regressors.shape) == 1:
                regressors = np.reshape(regressors, (-1, 1))
            beta = np.zeros([
                np.prod(saveshape[0:-1]), regressors.shape[1]+2])
        else:
            regressors = None
            beta = np.zeros([np.prod(saveshape[0:-1]), 2])

        filt = np.zeros((np.prod(saveshape[0:-1]), nt))
        sim = np.zeros_like(filt)
        residuals = np.zeros_like(filt)

        stdm = np.zeros((np.prod(saveshape[0:-1]), 1))
        stdp = np.zeros_like(stdm)
        r2 = np.zeros_like(stdm)
        r2_adj = np.zeros_like(stdm)
        exit = np.zeros_like(stdm)

        linearfit = odr.Model(self.multiplelinear)

        mag = np.reshape(mag, (-1, nt))
        ph = np.reshape(ph, (-1, nt))
        mm = np.mean(mag, axis=-1)
        mp = np.mean(ph, axis=-1)
        mask = mm > 0.03 * np.max(mm)

        for x in range(mag.shape[0]):
            if mask[x]:
                res, design = self.run_ODR(
                    mag[x, :], ph[x, :], regressors=regressors)
                est = res.y  # self.multiplelinear(res.beta, design)
                r2[x] = 1.0 - sum((mag[x, :] - est) ** 2) / \
                    sum((mag[x, :] - mm[x]) ** 2)
                r2_adj[x] = 1.0 - (1-r2[x])*(nt-1)/(nt-res.beta.size-2)
                exit[x] = res.info
                beta[x, :] = res.beta
                # take out scaled phase signal and re-mean may need correction
                sim[x, :] = est

                filt[x, :] = mag[x, :] - est + mm[x]
                # estimate residuals
                residuals[x, :] = np.sign(mag[x, :]-est)*(np.sum(res.delta**2,
                                                                 axis=0) + res.eps**2)

        _, outname, _ = split_filename(self.inputs.mag)
        print(outname)
        outnii = nb.Nifti1Image(np.reshape(sim, saveshape),
                                affine=f.affine, header=f.get_header())
        outnii.to_filename(outname + '_sim.nii.gz')
        outnii = nb.Nifti1Image(np.reshape(filt, saveshape), affine=f.affine,
                                header=f.get_header())
        outnii.to_filename(outname + '_filt.nii.gz')
        outnii = nb.Nifti1Image(np.reshape(residuals, saveshape),
                                affine=f.affine, header=f.get_header())
        outnii.to_filename(outname + '_residuals.nii.gz')

        # plot fit statistic info
        outnii = nb.Nifti1Image(np.reshape(stdp, saveshape[0:-1]),
                                affine=f.affine, header=f.get_header())
        outnii.to_filename(outname + '_stdp.nii.gz')
        outnii = nb.Nifti1Image(np.reshape(stdm, saveshape[0:-1]),
                                affine=f.affine, header=f.get_header())
        outnii.to_filename(outname + '_stdm.nii.gz')
        outnii = nb.Nifti1Image(np.reshape(r2, saveshape[0:-1]),
                                affine=f.affine, header=f.get_header())
        outnii.to_filename(outname + '_r2.nii.gz')
        outnii = nb.Nifti1Image(np.reshape(r2_adj, saveshape[0:-1]),
                                affine=f.affine, header=f.get_header())
        outnii.to_filename(outname + '_r2adj.nii.gz')
        outnii = nb.Nifti1Image(np.reshape(exit, saveshape[0:-1]),
                                affine=f.affine, header=f.get_header())
        outnii.to_filename(outname + '_exit.nii.gz')
        outnii = nb.Nifti1Image(np.reshape(beta, np.append(saveshape[0:-1], res.beta.size)),
                                affine=f.affine, header=f.get_header())
        outnii.to_filename(outname + '_beta.nii.gz')
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        _, base, _ = split_filename(self.inputs.mag)
        outputs["sim"] = os.path.abspath(base + '_sim.nii.gz')
        outputs["filt"] = os.path.abspath(base + '_filt.nii.gz')
        outputs["corr"] = os.path.abspath(base + '_r2.nii.gz')
        outputs["corradj"] = os.path.abspath(base + '_r2adj.nii.gz')
        outputs["residuals"] = os.path.abspath(base + '_residuals.nii.gz')
        outputs["stdm"] = os.path.abspath(base + '_stdm.nii.gz')
        outputs["stdp"] = os.path.abspath(base + '_stdp.nii.gz')
        outputs["exit"] = os.path.abspath(base + '_exit.nii.gz')
        outputs["beta"] = os.path.abspath(base + '_beta.nii.gz')
        return outputs
