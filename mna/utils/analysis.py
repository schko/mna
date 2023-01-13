import mne
import json
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt

def plot_source_time_course(ltc, orig_stc, label, mode, rel_mappings, xlim=None, ylim=None):
    if ylim is None:
        ylim = [-27, 22]
    if xlim is None:
        xlim = [[0, -1]]
    tc = ltc
    stc_label = orig_stc.in_label(label)
    fig, ax = plt.subplots(1)
    t = 1e3 * stc_label.times
    ax.plot(t, stc_label.data.T, 'k', linewidth=0.5, alpha=0.5)
    pe = [path_effects.Stroke(linewidth=5, foreground='w', alpha=0.5),
          path_effects.Normal()]
    ax.plot(t, tc[0], linewidth=3, label=mode, path_effects=pe)
    ax.legend(loc='upper right')
    ax.set(xlabel='Time (ms)', ylabel='Source amplitude',
           title='Activations in Label %r' % (rel_mappings[label.name]))
    # xlim=xlim, ylim=ylim)
    mne.viz.tight_layout()

def get_relevant_labels_mappings(path_to_base_package, regions_in_activity=None):
    """
    :param path_to_base_package:
    :return:
    reading the brodmann atlas mappings confirmed in freesurfer MNE template as mapping into these regions:
    http://www.brainm.com/software/pubs/dg/BA_10-20_ROI_Talairach/nearesteeg.htm
    """
    with open(f"{path_to_base_package}/data/annotated/brodmann_area_to_functional_name.json") as f:
        data = f.read()
    brodmann_mappings = json.loads(data, object_hook=lambda d: {int(k) if k.isdigit() else k: v for k, v in d.items()})
    if regions_in_activity is None:
        regions_in_activity = ['premotor', 'intermediate_frontal', 'dlpfc', 'frontopolar', 'ventral_anterior_cingulate',
                                'dorsal_anterior_cingulate','ventral_posterior_cingulate', 'dorsal_posterior_cingulate',
                                'superior_parietal','striate', 'parastriate','peristriate','middle_frontal']  # should map to the brodmann_area_to_func
        # json file and the corresponding brodmann_mappings dict
    elif regions_in_activity is 'all':
        regions_in_activity = list(brodmann_mappings.values()) # use all regions
    
    annot_labels = mne.read_labels_from_annot('fsaverage',parc='PALS_B12_Brodmann')[5:87] # only include the brodmann areas by num
    rel_labels = [] # relevant labels
    rel_mappings = {}
    for an_idx, annot in enumerate(annot_labels):
        area_num = int(annot.name.split('.')[1].split('-')[0])
        if area_num in brodmann_mappings and brodmann_mappings[area_num] in regions_in_activity:
            rel_labels.append(annot)
            rel_mappings[annot.name] = brodmann_mappings[area_num]
    return rel_labels, rel_mappings


def get_relevant_channels():
    rel_regions = {'premotor_regions': ['FC3', 'FC1', 'FCz', 'FC2', 'FC4'], 'dorsolateral_prefrontal': ['AF3', 'AFz', 'AF4'], 'intermediate_frontal': ['F3', 'F1', 'Fz', 'F2', 'F4']}
    all_regions = sum(rel_regions.values(),[])
    return rel_regions, all_regions