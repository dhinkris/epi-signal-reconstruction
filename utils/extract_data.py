import pandas as pd
import os
import nibabel as nib
import numpy as np

data=pd.read_csv('/data/mril/users/all/mrdata/research/processed/CNMC/chd_r01/fetal/wip/dhinesh/01242020/fetal_abcr01_34ven_01242020.csv')
abc="/data/mril/users/all/mrdata/research/processed/CNMC/chd_abc/fetal/wip/kk/recon/"
r01="/data/mril/users/all/mrdata/research/processed/CNMC/chd_r01/fetal/wip/kk/fetal_recon_kainz/"
outdir="/home/dhinesh/Desktop/brain_segmentation/multi-segmentation-3D-structural/"
for id, scan in zip(data['subject'], data['scan#']):
    if len(str(id))==3:
        scandir=os.path.join(abc, 'fetus_00'+str(id)+"/scan_0"+str(scan), 'files', 'recon' )
        if os.path.exists(os.path.join(scandir,'segmentations')):
            for segmentation in os.listdir(os.path.join(scandir,'segmentations')):
                if segmentation.endswith('final.nii.gz'):
                    segfile=os.path.join(os.path.join(scandir,'segmentations'), segmentation)
                    t2file=os.path.join(scandir, 'T2', os.path.basename(segfile).split('_tissue')[0]+'.nii.gz')
                    if os.path.exists(segfile) and os.path.exists(t2file):
                        seg_data=nib.load(segfile).get_data()
                        t2_data=nib.load(t2file).get_data()
                        if len(seg_data.shape)==4:
                            seg_data=seg_data[:,:,:,0]
                        print(os.path.join(outdir,'data', os.path.basename(segfile).split('_recon')[0]+'-label'))
                        np.save(os.path.join(outdir,'data', os.path.basename(segfile).split('recon')[0]+'-label'), seg_data)
                        if len(t2_data.shape)==4:
                            t2_data=t2_data[:,:,:,0]
                        np.save(os.path.join(outdir,'data', os.path.basename(segfile).split('recon')[0]+'-T2'), t2_data)

    elif len(str(id))==4:
        scandir=os.path.join(r01, 'fetus_0'+str(id)+"/scan_0"+str(scan), 'files', 'kainz')
        if os.path.exists(os.path.join(scandir,'segmentations')):
            for segmentation in os.listdir(os.path.join(scandir,'segmentations')):
                if segmentation.endswith('final.nii.gz'):
                    segfile=os.path.join(os.path.join(scandir,'segmentations'), segmentation)
                    t2file=os.path.join(scandir, 'T2', os.path.basename(segfile).split('_tissue')[0]+'.nii.gz')
                    if os.path.exists(segfile) and os.path.exists(t2file):
                        seg_data=nib.load(segfile).get_data()
                        t2_data=nib.load(t2file).get_data()
                        if len(seg_data.shape)==4:
                            seg_data=seg_data[:,:,:,0]
                        print(os.path.join(outdir,'data', os.path.basename(segfile).split('_recon')[0]+'-label'))
                        np.save(os.path.join(outdir,'data', os.path.basename(segfile).split('recon')[0]+'-label'), seg_data)
                        if len(t2_data.shape)==4:
                            t2_data=t2_data[:,:,:,0]
                        np.save(os.path.join(outdir,'data', os.path.basename(segfile).split('recon')[0]+'-T2'), t2_data)
