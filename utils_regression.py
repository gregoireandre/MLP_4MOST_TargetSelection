import os
import csv
import math
import copy
import time
import scipy
import itertools
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from colour import Color
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits
from astropy.stats import sigma_clipping, biweight, median_absolute_deviation
from sklearn.metrics import auc, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, Normalizer
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

#########General utilities functions################

def report2csv(model_index, all_models_path, catalog_filename, constraints, ann_parameters, regression_probem, zphot_safe_threshold, train_test_val_split, cv_fold_nbr, data_imputation, normalization, model_path, preprocessing, early_stopped_epoch, mae):

    csv_dict_inputs = {'model_index': [model_index],
                       'catalog': [catalog_filename],
                       'regression_problem': [regression_probem],
                       'zphot_safe_threshold': [zphot_safe_threshold],
                       'constraints': [constraints],
                       'data_imputation': [data_imputation],
                       'normalization': [normalization],
                       'train_test_val_split': [train_test_val_split],
                       'cv_fold_nbr': [cv_fold_nbr],
                       'preprocessing': [],
                       'preprocessing_params': []
                      }

    if preprocessing is not None:
        for idx, i in enumerate(preprocessing):
            if idx == 0:
                csv_dict_inputs['preprocessing'].append(i['method'])
                csv_dict_inputs['preprocessing_params'].append(str(i['arguments']))
            else:
                csv_dict_inputs['preprocessing'][0] += '___' + i['method']
                csv_dict_inputs['preprocessing_params'][0] += '___' + str(i['arguments'])
    else:
        csv_dict_inputs['preprocessing'].append(None)
        csv_dict_inputs['preprocessing_params'].append(None)

    csv_dict_performance = {'model_index': [model_index],
                            'early_stop': [early_stopped_epoch],
                            'mae': [mae]
                            }

    csv_dict_parameters = {'model_index': [model_index]}

    for i in list(ann_parameters.keys()):
        if (i != 'optimizer') and (i != 'kernel_regularizer'):
            csv_dict_parameters[i] = [ann_parameters[i]]

    benchmark_filename = os.path.join(all_models_path, 'Benchmark_ANN_inputs.csv')

    if os.path.isfile(benchmark_filename):
        csv_input = pd.read_csv(benchmark_filename)
        new_csv = pd.concat([csv_input, pd.DataFrame.from_dict(csv_dict_inputs)], sort=False)
        new_csv.to_csv(benchmark_filename, index=False)
    else:
        pd.DataFrame.from_dict(csv_dict_inputs).to_csv(benchmark_filename, index=False)

    benchmark_filename = os.path.join(all_models_path, 'Benchmark_ANN_parameters.csv')

    if os.path.isfile(benchmark_filename):
        csv_input = pd.read_csv(benchmark_filename)
        new_csv = pd.concat([csv_input, pd.DataFrame.from_dict(csv_dict_parameters)], sort=False)
        new_csv.to_csv(benchmark_filename, index=False)
    else:
        pd.DataFrame.from_dict(csv_dict_parameters).to_csv(benchmark_filename, index=False)

    benchmark_filename = os.path.join(all_models_path, 'Benchmark_ANN_performance.csv')

    if os.path.isfile(benchmark_filename):
        csv_input = pd.read_csv(benchmark_filename)
        new_csv = pd.concat([csv_input, pd.DataFrame.from_dict(csv_dict_performance)], sort=False)
        new_csv.to_csv(benchmark_filename, index=False)
    else:
        pd.DataFrame.from_dict(csv_dict_performance).to_csv(benchmark_filename, index=False)

    return

def read_fits(full_path):

    hdu = fits.open(full_path)
    data = hdu[1].data
    data_keys = hdu[1].columns.names

    nbr_features = len(data_keys)
    nbr_objects = len(data)
    data_array = np.zeros((nbr_objects, nbr_features), dtype=np.float64)

    for i in range(nbr_features):
        data_array[:,i] = data.field(i)

    return data_array, data_keys

def read_fits_as_dict(full_path):

    ####role####
    # Read a .fits file and returns the data and keys contained in the latter.

    ####inputs####
    # path : path of the .fits file to load
    # filename : filename of the .fits file to load

    ####outputs####
    # data_keys : The keys corresponding to the data inside the .fits file (eg DES_r, HSC_zphot, ...)
    # data : Object containing the data of the .fits file. The object has the same behavior than
    # python dictionnary (eg to access all the DES_r data just type data['DES_r'])

    hdu = fits.open(full_path)
    data = hdu[1].data
    data_keys = hdu[1].columns.names

    return data, data_keys

def csv2dict_list(csv_path):

    # Open variable-based csv, iterate over the rows and map values to a list of dictionaries containing key/value pairs

    with open(csv_path) as f:
        dict_list = [ {k:v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
    return dict_list

def get_model_weights_path(model_path, weights_flag):

    for file in os.listdir(model_path):
        if file.endswith(".hdf5"):
            final_model_weights = os.path.join(model_path, file)
            final_model_score = float((file.split('-')[1]).split('_')[-1])
            if weights_flag == 'final':
                return final_model_weights, final_model_score

    for file in os.listdir(os.path.join(model_path, 'checkpoints')):
        if file.endswith(".hdf5"):
            checkpoint_model_weights = os.path.join(model_path, 'checkpoints', file)
            checkpoint_score = float((file.split('-')[1]).split('_')[-1])
            if weights_flag == 'checkpoint':
                return checkpoint_model_weights

    if weights_flag == 'best':
        if checkpoint_score > final_model_score:
            return checkpoint_model_weights, checkpoint_score
        else:
            return final_model_weights, final_model_score

    return

def compute_polyfit_mean_errors(mags, errors, bin_width, polyfit_degree):

    ####role####
    # Given a set of magnitudes and errors, this function compute the mean error among bins of width bin_width for each magnitude.
    # A polyfit is then used to approximate the mean error in terms of magnitudes

    ####inputs####
    # mags (numpy array) : The set of magnitudes for which the polyfit has to be done. Given N samples and M magnitudes, the mags varibale should be an N x M array.
    # errors (numpy array) : The set of magnitudes errors for which the polyfit has to be done. Given N samples and M magnitudes, the mags varibale should be an N x M array. It should be ordered
    # in the same fashion than mags array.
    # bin_width (float) : The bin width to use for the mean error computation
    # polyfit_degree (int) : The degree of the polynomial function used to fit the mean error in terms of magnitudes
    # filename : filename of the .fits file to load

    ####outputs####
    # polyfit_coeff (list of numpy array) : The polynomial coefficient that approximate the mean error in terms of magnitude. The first element of the list
    # correspond to the coefficients related to the first magnitude of the mags/errors array.

    nbr_mags = mags.shape[1]
    polyfit_coeff = []
    for i in range(nbr_mags):
        nbr_bins = math.ceil(abs(np.amax(mags[:, i]) - np.amin(mags[:, i]))/bin_width)
        print(nbr_bins)
        print(np.amin(mags[:, i]))
        bins = [[np.amin(mags[:, i]) + j*bin_width, np.amin(mags[:, i]) + (j+1)*bin_width, 0, 0] for j in range(nbr_bins)]
        for idx, mag in enumerate(mags[:, i]):
            bin_number = math.floor((mag - np.amin(mags[:, i]))// bin_width)
            bins[bin_number][2] += 1
            bins[bin_number][3] += errors[idx,i]
        for k in range(len(bins)):
            if bins[k][2] > 0.0:
                bins[k][3] = bins[k][3]/bins[k][2]
        polyfit_x = []
        polyfit_y = []
        for h in range(len(bins)):
            if bins[h][2] > 0.0:
                polyfit_x.append((bins[h][1]+bins[h][0])/2.0)
                polyfit_y.append(bins[h][3])
        polyfit_y = np.log(polyfit_y)
        polyfit_coeff.append(np.polyfit(polyfit_x, polyfit_y, polyfit_degree))

    return polyfit_coeff

def plot_pdf(zphot_pdf, zphot, z_spec, savepath, title):

    start_bins = []
    stop_bins = []
    probability = []
    for i in range(len(zphot_pdf)):
        start_bins.append(zphot_pdf[i][0])
        stop_bins.append(zphot_pdf[i][1])
        probability.append(zphot_pdf[i][2])
    bins = start_bins + [stop_bins[-1]]
    probability = probability + [0]
    plt.figure(figsize=(19.2,10.8), dpi=100)
    # plt.step(bins, probability, color='red', label='zphot_PDF')
    plt.hist(bins, weights=probability, bins=len(bins)-1, rwidth=0.85, color='red')
    plt.axvline(x=z_spec, color='green', linestyle='-', label='z_spec')
    plt.axvline(x=zphot, color='black', linestyle=':', label='z_peak')
    ax = plt.gca()
    plt.xlabel('z')
    plt.ylabel('Probability')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(savepath, 'test_PDF_' + title + '.png'))
    plt.close()

    return

def compute_biweight_std(zphot_distribution, axis=None):

    zphot_bw_var = biweight.biweight_midvariance(zphot_distribution)
    zphot_bw_std = math.sqrt(zphot_bw_var)

    return zphot_bw_std

def compute_confidence_value(zphot_value_index, zphot_probs):

    conf = 0.0

    for i in range(zphot_value_index-3, zphot_value_index+4):
        if ( (i >= 0) and (i<len(zphot_probs)) ):
            conf += zphot_probs[i]
        else:
            continue
    if conf > 1.0:
        print('Final conf : ', conf)
        print(zphot_probs)
    return conf

def compute_residual(z_estimate, z_ref):

    residual = (z_estimate - z_ref)/(1.0 + z_ref)

    return residual

def compute_loss(z_estimate, z_ref, gamma=0.15):

    loss = 1.0 - 1.0/(1.0 + (compute_residual(z_estimate, z_ref)/gamma)**2)

    return loss

def compute_PDF_statistics(zphot, z_spec, bin_width):

    pdf_statistics = {
                        'bias' : [],
                        'mad' : 0.0,
                        'dispersion' : 0.0,
                        'outliers_rate' : 0.0,
                        'z_peak': {
                                    'value': 0.0,
                                    'risk': 0.0,
                                    'conf': 0.0
                                  },
                        'z_best': {
                                    'value': 0.0,
                                    'risk': 0.0,
                                    'conf': 0.0
                                  },
                        'z_mean': {
                                    'value': 0.0,
                                    'risk': 0.0,
                                    'conf': 0.0
                                  },
                        'z_median': {
                                        'value': None,
                                        'risk': 0.0,
                                        'conf': 0.0
                                    }
                     }
    sig_clip = sigma_clipping.SigmaClip(sigma=3, iters=3, cenfunc=biweight.biweight_location, stdfunc=compute_biweight_std)
    zphot_distribution_clipped = sig_clip(zphot)
    zphot_distribution_clipped = zphot_distribution_clipped.compressed()

    zphot_nbr_bins = math.ceil(abs(np.amax(zphot_distribution_clipped) - np.amin(zphot_distribution_clipped))/bin_width) + 1
    zphot_pdf = [[np.amin(zphot_distribution_clipped) + j*bin_width, np.amin(zphot_distribution_clipped) + (j+1)*bin_width, 0.0] for j in range(zphot_nbr_bins)]
    for idxz, z in enumerate(zphot_distribution_clipped):
        bin_number = math.floor((z - np.amin(zphot_distribution_clipped))// bin_width)
        zphot_pdf[bin_number][2] += 1

    pdf_statistics['mad'] = median_absolute_deviation(pdf_statistics['bias'])
    pdf_statistics['dispersion'] = 1.48*pdf_statistics['mad']
    for i in zphot_distribution_clipped:
        pdf_statistics['bias'].append(compute_residual(i, z_spec))
    abs_bias = np.array([abs(k) for k in pdf_statistics['bias']])
    pdf_statistics['outliers_rate'] = len(np.where(abs_bias > 0.15)[0])/len(abs_bias)

    zphot_values, zphot_counts = np.unique(zphot_distribution_clipped, return_counts=True)
    zphot_probs = zphot_counts/np.sum(zphot_counts)

    # Compute zphot_peak, we have to handle the case where the highest probability correspond to several bins.
    # In this case, the zphot_peak is taken as the mean of all the zphots corresponding to highest probability peaks
    indexes_max_prob = np.where(zphot_probs == np.amax(zphot_probs))[0]
    zphot_peak_tmp = 0.0
    for k in indexes_max_prob:
        zphot_peak_tmp += zphot_values[k]
    zphot_peak_tmp = zphot_peak_tmp/(len(indexes_max_prob))
    pdf_statistics['z_peak']['value'] = zphot_peak_tmp
    zphot_peak_pdf_idx = math.floor((pdf_statistics['z_peak']['value'] -  zphot_pdf[0][0])//bin_width)

    zphot_risk = []
    probs_sum = 0.0
    for i in range(zphot_values.shape[0]):
        risk = 0.0
        probs_sum += zphot_probs[i]
        if (probs_sum >= 0.5) and (pdf_statistics['z_median']['value'] is None):
            pdf_statistics['z_median']['value'] = zphot_values[i]
        pdf_statistics['z_mean']['value'] +=  zphot_probs[i]*zphot_values[i]
        for j in range(zphot_values.shape[0]):
            risk += zphot_probs[j]*compute_loss(zphot_values[i], zphot_values[j])
        zphot_risk.append(risk)

    zphot_peak_idx = (np.abs(zphot_values - pdf_statistics['z_peak']['value'])).argmin()
    zphot_best_idx = np.argmin(zphot_risk)
    zphot_mean_idx = (np.abs(zphot_values - pdf_statistics['z_mean']['value'])).argmin()
    zphot_median_idx = (np.abs(zphot_values - pdf_statistics['z_median']['value'])).argmin()

    pdf_statistics['z_best']['risk'] = np.amin(zphot_risk)
    pdf_statistics['z_peak']['risk'] = zphot_risk[zphot_peak_idx]
    pdf_statistics['z_best']['value'] = zphot_values[zphot_best_idx]
    pdf_statistics['z_peak']['value'] = zphot_peak_tmp
    pdf_statistics['z_peak']['conf'] = compute_confidence_value(zphot_peak_idx, zphot_probs)
    pdf_statistics['z_best']['conf'] = compute_confidence_value(zphot_best_idx, zphot_probs)
    pdf_statistics['z_mean']['conf'] = compute_confidence_value(zphot_mean_idx, zphot_probs)
    pdf_statistics['z_median']['conf'] = compute_confidence_value(zphot_median_idx, zphot_probs)

    for j in range(zphot_values.shape[0]):
        pdf_statistics['z_mean']['risk'] += zphot_probs[j]*compute_loss(pdf_statistics['z_mean']['value'], zphot_values[j])
        pdf_statistics['z_median']['risk'] += zphot_probs[j]*compute_loss(pdf_statistics['z_median']['value'], zphot_values[j])

    return pdf_statistics, zphot_pdf, zphot, zphot_peak_pdf_idx

# def compute_PDF_statistics(zphot_pdf, zphot, z_spec, bin_width):
#
#     zphot = list(zphot)
#     pdf_statistics = {
#                         'bias' : [],
#                         'mad' : 0.0,
#                         'dispersion' : 0.0,
#                         'outliers_rate' : 0.0,
#                         'z_peak': {
#                                     'value': 0.0,
#                                     'risk': 0.0,
#                                     'conf': 0.0
#                                   },
#                         'z_best': {
#                                     'value': 0.0,
#                                     'risk': 0.0,
#                                     'conf': 0.0
#                                   },
#                         'z_mean': {
#                                     'value': 0.0,
#                                     'risk': 0.0,
#                                     'conf': 0.0
#                                   },
#                         'z_median': {
#                                         'value': None,
#                                         'risk': 0.0,
#                                         'conf': 0.0
#                                     }
#                      }
#     zphot_distribution = []
#     for idx, i in enumerate(zphot_pdf):
#         zphot_distribution += [(i[1] + i[0])/2.0]*int(i[2])
#     sig_clip = sigma_clipping.SigmaClip(sigma=3, iters=3, cenfunc=biweight.biweight_location, stdfunc=compute_biweight_std)
#     zphot_distribution_clipped = sig_clip(zphot_distribution)
#     zphot_distribution_clipped = zphot_distribution_clipped.compressed()
#
#     zphot_nbr_bins = math.ceil(abs(np.amax(zphot_distribution_clipped) - np.amin(zphot_distribution_clipped))/bin_width) + 1
#     zphot_pdf = [[np.amin(zphot_distribution_clipped) + j*bin_width, np.amin(zphot_distribution_clipped) + (j+1)*bin_width, 0.0] for j in range(zphot_nbr_bins)]
#     for idxz, z in enumerate(zphot_distribution_clipped):
#         bin_number = math.floor((z - np.amin(zphot_distribution_clipped))// bin_width)
#         zphot_pdf[bin_number][2] += 1
#
#     idx_to_remove = []
#     for idxz in range(len(zphot)):
#         if (zphot[idxz] < np.amin(zphot_distribution_clipped)) or (zphot[idxz] < np.amin(zphot_distribution_clipped)) :
#             idx_to_remove.append(idxz)
#     idx_to_remove.sort(reverse=True)
#     for k in idx_to_remove:
#         del zphot[k]
#
#     zphot = np.array(zphot)
#     # print('zphot_distribution_clipped : ', len(zphot_distribution_clipped), zphot_distribution_clipped)
#     # print('zphot_pdf_cleaned : ', len(zphot_pdf), zphot_pdf)
#     # print('zphot_cleaned : ', len(zphot), zphot)
#
#     pdf_statistics['mad'] = median_absolute_deviation(pdf_statistics['bias'])
#     pdf_statistics['dispersion'] = 1.48*pdf_statistics['mad']
#     for i in zphot_distribution_clipped:
#         pdf_statistics['bias'].append(compute_residual(i, z_spec))
#     abs_bias = np.array([abs(k) for k in pdf_statistics['bias']])
#     pdf_statistics['outliers_rate'] = len(np.where(abs_bias > 0.15)[0])/len(abs_bias)
#
#     zphot_values, zphot_counts = np.unique(zphot_distribution_clipped, return_counts=True)
#     zphot_probs = zphot_counts/np.sum(zphot_counts)
#
#     # Compute zphot_peak, we have to handle the case where the highest probability correspond to several bins.
#     # In this case, the zphot_peak is taken as the mean of all the zphots corresponding to highest probability peaks
#     indexes_max_prob = np.where(zphot_probs == np.amax(zphot_probs))[0]
#     zphot_peak_tmp = 0.0
#     for k in indexes_max_prob:
#         zphot_peak_tmp += zphot_values[k]
#     zphot_peak_tmp = zphot_peak_tmp/(len(indexes_max_prob))
#     pdf_statistics['z_peak']['value'] = zphot_peak_tmp
#     zphot_peak_pdf_idx = math.floor((pdf_statistics['z_peak']['value'] -  zphot_pdf[0][0])//bin_width)
#     # print(zphot_peak_pdf_idx)
#     # print(len(zphot_pdf))
#     # print(zphot_pdf[-1][:])
#     # print(pdf_statistics['z_peak']['value'])
#     # print(zphot_pdf[zphot_peak_pdf_idx][:])
#
#     zphot_risk = []
#     probs_sum = 0.0
#     for i in range(zphot_values.shape[0]):
#         risk = 0.0
#         probs_sum += zphot_probs[i]
#         if (probs_sum >= 0.5) and (pdf_statistics['z_median']['value'] is None):
#             pdf_statistics['z_median']['value'] = zphot_values[i]
#         pdf_statistics['z_mean']['value'] +=  zphot_probs[i]*zphot_values[i]
#         for j in range(zphot_values.shape[0]):
#             risk += zphot_probs[j]*compute_loss(zphot_values[i], zphot_values[j])
#         zphot_risk.append(risk)
#
#     zphot_peak_idx = (np.abs(zphot_values - pdf_statistics['z_peak']['value'])).argmin()
#     zphot_best_idx = np.argmin(zphot_risk)
#     zphot_mean_idx = (np.abs(zphot_values - pdf_statistics['z_mean']['value'])).argmin()
#     zphot_median_idx = (np.abs(zphot_values - pdf_statistics['z_median']['value'])).argmin()
#
#     pdf_statistics['z_best']['risk'] = np.amin(zphot_risk)
#     pdf_statistics['z_peak']['risk'] = zphot_risk[zphot_peak_idx]
#     pdf_statistics['z_best']['value'] = zphot_values[zphot_best_idx]
#     pdf_statistics['z_peak']['value'] = zphot_peak_tmp
#     pdf_statistics['z_peak']['conf'] = compute_confidence_value(zphot_peak_idx, zphot_probs)
#     pdf_statistics['z_best']['conf'] = compute_confidence_value(zphot_best_idx, zphot_probs)
#     pdf_statistics['z_mean']['conf'] = compute_confidence_value(zphot_mean_idx, zphot_probs)
#     pdf_statistics['z_median']['conf'] = compute_confidence_value(zphot_median_idx, zphot_probs)
#
#     for j in range(zphot_values.shape[0]):
#         pdf_statistics['z_mean']['risk'] += zphot_probs[j]*compute_loss(pdf_statistics['z_mean']['value'], zphot_values[j])
#         pdf_statistics['z_median']['risk'] += zphot_probs[j]*compute_loss(pdf_statistics['z_median']['value'], zphot_values[j])
#
#     return pdf_statistics, zphot_pdf, zphot, zphot_peak_pdf_idx

def compute_PDF(normalization, normalizer, data_imputation_values, test_data_id, test_data, test_label, model, nbr_samples, alpha, polyfit_degree, weight_threshold):

    ####role####
    # This function compute Probability Density Function instead of a single point estimate in the framework of redshift estimation.
    # It is the implementation of the bimodal polynomial method of METAPHOR paper which is described in the report and available on arXiv :

    # Algorithm main steps :
    # For each of the sample for which the redshift has to be estimated
    # 1) Generate nbr_samples from the single sample of step 1 by introdution of gaussian noise with variance proportionnal to the error on the magnitude
    # 2) Estimate the redshift for each of the (nbr_samples + 1) samples
    # 3) Given a bin width for the redshift, compute the Probability Density Function from the (nbr_samples + 1) predictions

    ####inputs####
    # normalization (string): The type of normalization used during training of the model
    # normalizer (various):
    # data_imputation_values (numpy array): Array containing data imputation value used by model for each features
    # test_data_id (numpy array): Single column numpy array containing DES_id for each sample
    # test_data (numpy array): The set of sample o which PDF estimation will be computed
    # test_label (numpy array): If available, the true redshifts for each sample of test_data. The true redshifts are only used to evaluate performances of the PDF computation, if the true redshifts are not available,
    # one should give dummy z_spec value in the for of a numpy array with same shape than test_data_id
    # model (keras/sklearn object): The keras/sklearn regressor that will be used to evaluate the redshift
    # nbr_samples: The number of sample to generate for each of the test sample
    # alpha (float): a multiplicative constant used in the data augmentation process
    # polyfit_degree (int): The degree of the polyfit used to estimate the mean error in terms of magnitudes (see compute_polyfit_mean_errors function)
    # weight_threshold (float): The threshold to use during weight computation

    ####outputs####
    # zphot_peak (list) : Highest probability redshift estimate extracted from PDF of each test_data sample
    # zphot_pdf :
    # metaphor_statistics_dict (dict): A dictionnary containing statistics relative to PDFs performances. Only relevant if true test_label are provided.

    ####Note###
    # In order to reduce the effect of missing values on the PDF estimation, the data augmentation with gaussian noise is only performed on available magnitudes.
    # After data augmentation, the magnitudes that were not available are imputed with values stored in data_imputation_values variable.
    # For the sake of simplicity and efficiency, only two case were considered here :
    # 1) Only DES magnitudes available
    # 2) All magnitudes available (DES, VHS, WISE)
    # As polyfit does not handle missing values, it is performed for the two scenarios (only on DES and on all magnitudes)
    # The flag 'tot' in the below variables is used to represent the variable relative to


    nbr_objects = test_data.shape[0]
    mags = test_data[:, 0:5]
    mags_tot = test_data[:, 0:8]
    errors = test_data[:, 8:13]
    errors_tot = test_data[:, 8:]
    nbr_mags = mags.shape[1]
    nbr_mags_tot = int(test_data.shape[1]/2)
    metaphor_statistics_dict = {
                                   'DES_id': [],
                                   'perc_0.05': [],
                                   'perc_0.15': [],
                                   'bias': [],
                                   'stdev': [],
                                   'NMAD': [],
                                   'perc_outliers': [],
                                   'skew': [],
                                   'perc_peak': 0.0,
                                   'perc_onebin': 0.0,
                                   'perc_inpdf': 0.0,
                                   'perc_outpdf': 0.0
                               }
    HSC_statistics_dict = {
                            'z_peak': [],
                            'risk_peak': [],
                            'conf_peak': [],
                            'z_best': [],
                            'risk_best': [],
                            'conf_best': [],
                            'z_mean': [],
                            'risk_mean': [],
                            'conf_mean': [],
                            'z_median': [],
                            'risk_median': [],
                            'conf_median': [],
                            'mad': [],
                            'dispersion': [],
                            'outliers_rate': []
                          }

    zphot_pdf = []
    not_nan_idx = [i for i in range(nbr_objects) if not any(np.isnan(test_data[i,:]))]
    polyfit_errors_coeff = compute_polyfit_mean_errors(mags, errors, 0.05, polyfit_degree)
    polyfit_errors_coeff_tot = compute_polyfit_mean_errors(mags_tot[not_nan_idx, :], errors_tot[not_nan_idx, :], 0.05, polyfit_degree)

    for i in range(nbr_objects):
        print('iteration ', i+1 , ' / ', nbr_objects, end="\r")
        # print('iteration ', i+1 , ' / ', nbr_objects)

        augmented_object = np.zeros((nbr_samples + 1, nbr_mags_tot*2))
        augmented_object[0,:] = test_data[i, :]

        if i in not_nan_idx:
            for j in range(nbr_mags_tot):
                weight_coefficient = 0.0
                for idxk, k in enumerate(polyfit_errors_coeff_tot[j]):
                    weight_coefficient += k*pow(test_data[i, j], polyfit_degree-idxk)
                weight_coefficient = math.exp(weight_coefficient)
                if weight_coefficient < weight_threshold:
                    weight_coefficient = weight_threshold
                augmented_object[1:,j] = test_data[i, j] + alpha*weight_coefficient*np.random.normal(0.0, 1.0, nbr_samples)
            augmented_object[:,nbr_mags:2*nbr_mags_tot] = np.tile(test_data[i, nbr_mags:2*nbr_mags_tot], (nbr_samples + 1, 1))
        else:
            for j in range(nbr_mags):
                weight_coefficient = 0.0
                for idxk, k in enumerate(polyfit_errors_coeff[j]):
                    weight_coefficient += k*pow(test_data[i, j], polyfit_degree-idxk)
                weight_coefficient = math.exp(weight_coefficient)
                if weight_coefficient < weight_threshold:
                    weight_coefficient = weight_threshold
                augmented_object[1:,j] = test_data[i, j] + alpha*weight_coefficient*np.random.normal(0.0, 1.0, nbr_samples)
            augmented_object[:,nbr_mags:2*nbr_mags_tot] = np.tile(test_data[i, nbr_mags:2*nbr_mags_tot], (nbr_samples + 1, 1))

        if normalization:
            augmented_object = apply_normalization(augmented_object, normalization, normalizer)

        augmented_object = fill_empty_entries(augmented_object, data_imputation_values)

        zphot = model.predict(augmented_object)
        zphot = np.squeeze(zphot)
        zphot_bin_width = 0.01

        pdf_statistics, zphot_bins_clipped, zphot_clipped, zpeak_idx = compute_PDF_statistics(zphot, test_label[i], zphot_bin_width)

        HSC_statistics_dict['z_best'].append(float(pdf_statistics['z_best']['value']))
        HSC_statistics_dict['risk_best'].append(float(pdf_statistics['z_best']['risk']))
        HSC_statistics_dict['conf_best'].append(float(pdf_statistics['z_best']['conf']))
        HSC_statistics_dict['z_mean'].append(float(pdf_statistics['z_mean']['value']))
        HSC_statistics_dict['risk_mean'].append(float(pdf_statistics['z_mean']['risk']))
        HSC_statistics_dict['conf_mean'].append(float(pdf_statistics['z_mean']['conf']))
        HSC_statistics_dict['z_median'].append(float(pdf_statistics['z_median']['value']))
        HSC_statistics_dict['risk_median'].append(float(pdf_statistics['z_median']['risk']))
        HSC_statistics_dict['conf_median'].append(float(pdf_statistics['z_median']['conf']))
        HSC_statistics_dict['z_peak'].append(float(pdf_statistics['z_peak']['value']))
        HSC_statistics_dict['risk_peak'].append(float(pdf_statistics['z_peak']['risk']))
        HSC_statistics_dict['conf_peak'].append(float(pdf_statistics['z_peak']['conf']))
        HSC_statistics_dict['mad'].append(float(pdf_statistics['mad']))
        HSC_statistics_dict['dispersion'].append(float(pdf_statistics['dispersion']))
        HSC_statistics_dict['outliers_rate'].append(float(pdf_statistics['outliers_rate']))

        nbr_samples_clipped = 0
        for k in zphot_bins_clipped:
            nbr_samples_clipped += k[2]

        # Compute probability from bins count and store bins and corresponding probability in zphot_pdf variable
        for k in range(len(zphot_bins_clipped)):
            zphot_bins_clipped[k][2] = zphot_bins_clipped[k][2]/nbr_samples_clipped
        zphot_pdf.append(zphot_bins_clipped)

        # Convert zphot_bins_clipped to np.array for statistic PDF derivation
        zphot_bins_clipped = np.array(zphot_bins_clipped)

        if ((zphot_bins_clipped[zpeak_idx,0] <= test_label[i]) and (test_label[i] < zphot_bins_clipped[zpeak_idx,1])):
            metaphor_statistics_dict['perc_peak'] += 1
        elif ( (zpeak_idx - 1 >= 0) and (zpeak_idx + 1 < zphot_bins_clipped.shape[0]) ):
            if ((zphot_bins_clipped[zpeak_idx - 1,0] <= test_label[i]) and (test_label[i] < zphot_bins_clipped[zpeak_idx - 1,1])) or ((zphot_bins_clipped[zpeak_idx+1,0] <= test_label[i]) and (test_label[i] < zphot_bins_clipped[zpeak_idx+1,1])):
                metaphor_statistics_dict['perc_onebin'] += 1
            elif ((zphot_bins_clipped[0,0] <= test_label[i]) and (test_label[i] < zphot_bins_clipped[-1,1])):
                metaphor_statistics_dict['perc_inpdf'] += 1
            else:
                metaphor_statistics_dict['perc_outpdf'] += 1
        elif ((zpeak_idx - 1 < 0) and (zpeak_idx + 1 < zphot_bins_clipped.shape[0])):
            if ((zphot_bins_clipped[zpeak_idx+1,0] <= test_label[i]) and (test_label[i] < zphot_bins_clipped[zpeak_idx+1,1])):
                metaphor_statistics_dict['perc_onebin'] += 1
            elif ((zphot_bins_clipped[0,0] <= test_label[i]) and (test_label[i] < zphot_bins_clipped[-1,1])):
                metaphor_statistics_dict['perc_inpdf'] += 1
            else:
                metaphor_statistics_dict['perc_outpdf'] += 1
        elif (zpeak_idx + 1 >= zphot_bins_clipped.shape[0]):
            if ((zphot_bins_clipped[zpeak_idx - 1,0] <= test_label[i]) and (test_label[i] < zphot_bins_clipped[zpeak_idx - 1,1])):
                metaphor_statistics_dict['perc_onebin'] += 1
            elif ((zphot_bins_clipped[0,0] <= test_label[i]) and (test_label[i] < zphot_bins_clipped[-1,1])):
                metaphor_statistics_dict['perc_inpdf'] += 1
            else:
                metaphor_statistics_dict['perc_outpdf'] += 1


        residuals = []
        for k in range(len(zphot_clipped)):
            residuals.append((test_label[i]-zphot_clipped[k])/(1+test_label[i]))
        residuals = np.array(residuals)

        metaphor_statistics_dict['DES_id'].append(test_data_id[i])
        metaphor_statistics_dict['bias'].append(np.mean(residuals))
        metaphor_statistics_dict['stdev'].append(np.std(residuals))
        metaphor_statistics_dict['NMAD'].append(np.median(abs(residuals)))
        metaphor_statistics_dict['perc_outliers'].append(len(np.where(abs(residuals) > 0.15)[0])/residuals.shape[0])
        metaphor_statistics_dict['perc_0.05'].append(len(np.where(abs(residuals) < 0.05)[0])/residuals.shape[0])
        metaphor_statistics_dict['perc_0.15'].append(len(np.where(abs(residuals) <= 0.15)[0])/residuals.shape[0])
        metaphor_statistics_dict['skew'].append(scipy.stats.skew(residuals, axis=0))

    metaphor_statistics_dict['perc_peak'] = metaphor_statistics_dict['perc_peak']/nbr_objects
    metaphor_statistics_dict['perc_onebin'] = metaphor_statistics_dict['perc_onebin']/nbr_objects
    metaphor_statistics_dict['perc_inpdf'] = metaphor_statistics_dict['perc_inpdf']/nbr_objects
    metaphor_statistics_dict['perc_outpdf'] = metaphor_statistics_dict['perc_outpdf']/nbr_objects


    return zphot_pdf, HSC_statistics_dict, metaphor_statistics_dict

def plot_regression_scatterplot(Y_pred, Y_val, mean_square_error, savepath):

    max_zphot = np.amax(Y_pred)
    nbr_bins = math.ceil((np.amax(Y_pred) - np.amin(Y_pred))/0.2)

    plt.figure(figsize=(19.2,10.8), dpi=100)
    plt.plot([0,max_zphot], [0,max_zphot], linestyle='-', lw=1, color='red')
    plt.scatter(Y_val, Y_pred, color='black', marker='X', s=1)
    ax = plt.gca()
    ax.set_ylim(bottom=0.0, top=max_zphot)
    ax.set_xlim(left=0.0, right=max_zphot)
    plt.xlabel('z_spec')
    plt.ylabel('z_phot')
    plt.locator_params(axis='y', nbins=nbr_bins)
    plt.locator_params(axis='x', nbins=nbr_bins)
    plt.title('Photometric Redshift Regression : mae = {:0.2f}'.format(mean_square_error))
    plt.savefig(os.path.join(savepath, 'Z_scatter_plot.png'))
    plt.close()

    return

def plot_histogram_distribution(Y_pred, Y_val, mean_square_error, savepath):

    z_spec_min = np.amin(Y_val)
    z_spec_max = np.amax(Y_val)
    nbr_bins = math.ceil((np.amax(Y_val) - np.amin(Y_val))/0.2)

    plt.figure(figsize=(19.2,10.8), dpi=100)
    plt.hist(Y_val, bins=nbr_bins, density=True, histtype='step', fill=False, color='green')
    plt.hist(Y_pred, bins=nbr_bins, density=True, range=(z_spec_min, z_spec_max), histtype='step', fill=False, color='red')
    ax = plt.gca()
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    plt.xlabel('z_spec')
    plt.ylabel('Normalized Density')
    plt.locator_params(axis='y', nbins=nbr_bins)
    plt.locator_params(axis='x', nbins=nbr_bins)
    plt.title('Photometric Redshift Regression : mae = {:0.2f}'.format(mean_square_error))
    plt.savefig(os.path.join(savepath, 'Z_density_plot.png'))
    plt.close()

    return

def heatmap_density_plot(Y_pred, Y_val, mean_square_error, savepath):

    max_zphot = np.amax(Y_pred)
    nbr_bins = math.ceil((np.amax(Y_pred) - np.amin(Y_pred))/0.2)

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    nbins=300
    xy = np.vstack([Y_val,Y_pred])
    z = gaussian_kde(xy)(xy)

    idx = z.argsort()
    Y_val, Y_pred, z = Y_val[idx], Y_pred[idx], z[idx]

    fig = plt.figure()
    img = plt.scatter(Y_val, Y_pred, c=z, s=5, edgecolor='', cmap=plt.cm.get_cmap('jet'))
    ax = plt.gca()
    ax.set_ylim(bottom=0.0, top=max_zphot)
    ax.set_xlim(left=0.0, right=max_zphot)
    fig.colorbar(img, ax=ax)
    plt.xlabel('z_spec')
    plt.ylabel('z_phot')
    plt.locator_params(axis='y', nbins=nbr_bins)
    plt.locator_params(axis='x', nbins=nbr_bins)
    plt.title('Photometric Redshift Regression : mae = {:0.2f}'.format(mean_square_error))
    plt.savefig(os.path.join(savepath, 'Z_heatmap_scatter_plot.png'))
    plt.show()

    return

def compute_PDF_probability(pdf, lower_range, upper_range):
    probability_range = 0.0
    z_peaks = []

    for i in range(len(pdf)):

        if (pdf[i][0] >= lower_range) and (pdf[i][0] <= upper_range):
            probability_range += pdf[i][2]

    return probability_range

def compute_PDF_variance(pdf):
    probabilities = []
    z_peaks = []

    for i in range(len(pdf)):
        probabilities.append(pdf[i][2])
        z_peaks.append((pdf[i][1]+pdf[i][0])/2.0)

    z_peaks_average = np.average(z_peaks, weights=probabilities)
    # Fast and numerically precise:
    variance = math.sqrt(np.average((z_peaks-z_peaks_average)**2, weights=probabilities))

    return variance

#######Dataset processing methods#########

def load_dataset(dataset_path, model_path, regression_problem, zphot_safe_threshold, train_test_val_split, cv_fold_nbr, normalization, data_imputation, random_seed):

    training_dataset_filename = regression_problem + '_' + str(zphot_safe_threshold) + '-zsafe' + '_train_' + str(train_test_val_split[0]) + '_' + str(train_test_val_split[1]) + '_' + str(cv_fold_nbr) + '.fits'
    validation_dataset_filename = regression_problem + '_' + str(zphot_safe_threshold) + '-zsafe' + '_val_' + str(train_test_val_split[0]) + '_' + str(train_test_val_split[1]) + '_' + str(cv_fold_nbr) + '.fits'
    testing_dataset_filename = regression_problem + '_' + str(zphot_safe_threshold) + '-zsafe' + '_test_' + str(train_test_val_split[0]) + '_' + str(train_test_val_split[1]) + '.fits'

    # load  dataset
    training_dataset, data_keys = read_fits(os.path.join(dataset_path, training_dataset_filename))
    np.random.shuffle(training_dataset)
    validation_dataset, _ = read_fits(os.path.join(dataset_path, validation_dataset_filename))
    np.random.shuffle(validation_dataset)
    testing_dataset, _ = read_fits(os.path.join(dataset_path, testing_dataset_filename))
    np.random.shuffle(testing_dataset)

    data_keys = data_keys[1:]

    # split into input (X) and output (Y) variables
    DES_id_train = training_dataset[:,0]
    sample_weights_train = training_dataset[:,1]
    data_keys_train = data_keys[1:-1]
    X_train = training_dataset[:,2:-1]
    Y_train = training_dataset[:,-1]

    save_training_data(X_train, data_keys_train, model_path)

    # split into input (X) and output (Y) variables
    DES_id_val = validation_dataset[:,0]
    sample_weights_val = validation_dataset[:,1]
    X_val = validation_dataset[:,2:-1]
    Y_val = validation_dataset[:,-1]

    # split into input (X) and output (Y) variables
    DES_id_test = testing_dataset[:,0]
    sample_weights_test = testing_dataset[:,1]
    X_test = testing_dataset[:,2:-1]
    Y_test = testing_dataset[:,-1]

    sample_weights_train, sample_weights_val, sample_weights_test = process_sample_weights(sample_weights_train, sample_weights_val, sample_weights_test, zphot_safe_threshold)

    return DES_id_train, X_train, Y_train, sample_weights_train, DES_id_val, X_val, Y_val, sample_weights_val, DES_id_test, X_test, Y_test, sample_weights_test, data_keys

def apply_normalization(dataset, normalization, normalizer):

    if normalization == 'cr':
        dataset_norm = apply_reduce_center(dataset, normalizer)
        return dataset_norm
    elif normalization == 'quant':
        dataset_norm  = apply_quantile_transform(dataset, normalizer)
        return dataset_norm
    else:
        return dataset

def apply_reduce_center(dataset, mean_std):

    dataset_cr = dataset.copy()

    for i in range(mean_std.shape[0]):
        dataset_cr[:,i] = (dataset_cr[:,i] - mean_std[i, 0])/mean_std[i, 1]

    return dataset_cr

def apply_quantile_transform(dataset, quantile_transformer):

    dataset_quant = dataset.copy()
    dataset_quant = quantile_transformer.transform(dataset_quant)

    return dataset_quant

def fill_empty_entries(dataset, data_imputation_values):

    nbr_features = dataset.shape[1]
    dataset_filled = dataset.copy()

    for i in range(nbr_features):
        nan_mask = np.isnan(dataset[:,i])
        nan_idx = np.where(nan_mask)[0]
        dataset_filled[nan_idx,i] = data_imputation_values[i]

    return dataset_filled

def normalize_dataset(X_train, X_val, X_test, normalization, random_seed):

    if normalization == 'cr':
        X_train_norm, X_val_norm, X_test_norm, mean_std = reduce_and_center_dataset(X_train, X_val, X_test)
        return X_train_norm, X_val_norm, X_test_norm, mean_std
    elif normalization == 'quant':
        X_train_norm, X_val_norm, X_test_norm = quantile_transform_dataset(X_train, X_val, X_test, random_seed)
        return X_train_norm, X_val_norm, X_test_norm, quantile_transformer
    else:
        return X_train, X_val, X_test, None

def reduce_and_center_dataset(X_train, X_val, X_test):

    nbr_object = X_train.shape[0]
    nbr_features = X_train.shape[1]
    mean_std = np.zeros((nbr_features, 2))
    X_train_cr = X_train.copy()
    X_val_cr = X_val.copy()
    X_test_cr = X_test.copy()

    for i in range(nbr_features):
        nan_mask = np.isnan(X_train[:,i])
        idx = np.where(~nan_mask)[0]
        mean_std[i, 0] = np.mean(X_train[idx,i])
        mean_std[i, 1] = np.std(X_train[idx, i])
        print(i, mean_std[i, 0])
        print(i, mean_std[i, 1])
        X_train_cr[:,i] = (X_train_cr[:,i] - mean_std[i, 0])/mean_std[i, 1]
        X_val_cr[:,i] = (X_val_cr[:,i] - mean_std[i, 0])/mean_std[i, 1]
        X_test_cr[:,i] = (X_test_cr[:,i] - mean_std[i, 0])/mean_std[i, 1]

    return X_train_cr, X_val_cr, X_test_cr, mean_std

def quantile_transform_dataset(X_train, X_val, X_test, random_seed):

    X_train_quant = X_train.copy()
    X_val_quant = X_val.copy()
    X_test_quant = X_test.copy()

    quantile_transformer = QuantileTransformer(random_state=random_seed)
    X_train_quant = quantile_transformer.fit_transform(X_train_quant)
    X_val_quant = quantile_transformer.transform(X_val_quant)
    X_test_quant = quantile_transformer.transform(X_test_quant)

    return X_train_quant, X_val_quant, X_test_quant, quantile_transformer

def fill_empty_entries_dataset(X_train, X_val, X_test, data_imputation, constraints):

    nbr_object = X_train.shape[0]
    nbr_features = X_train.shape[1]
    X_train_filled = X_train.copy()
    X_val_filled = X_val.copy()
    X_test_filled = X_test.copy()

    if data_imputation == 'max':

        max_features = np.zeros(nbr_features)

        if 'color' in constraints:

            for i in range((nbr_features-1),-1,-1):
                if i ==0:
                    max_features[i] = math.ceil(max(np.nanmax(X_train[:,6]) - np.nanmax(X_train[:,7]), np.nanmax(X_val[:,6] - X_val[:,7]), np.nanmax(X_test[:,6] - X_test[:,7])))
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = max_features[i]
                elif i ==1:
                    max_features[i] = math.ceil(max(np.nanmax(X_train[:,7]) - np.nanmax(X_train[:,8]), np.nanmax(X_val[:,7] - X_val[:,8]), np.nanmax(X_test[:,7] - X_test[:,8])))
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = max_features[i]
                elif i==2:
                    max_features[i] = math.ceil(max(np.nanmax(X_train[:,8]) - np.nanmax(X_train[:,9]), np.nanmax(X_val[:,8] - X_val[:,9]), np.nanmax(X_test[:,8] - X_test[:,9])))
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = max_features[i]
                elif i==3:
                    max_features[i] = math.ceil(max(np.nanmax(X_train[:,9]) - np.nanmax(X_train[:,10]), np.nanmax(X_val[:,9] - X_val[:,10]), np.nanmax(X_test[:,9] - X_test[:,10])))
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = max_features[i]
                elif i==4:
                    max_features[i] = math.ceil(max(np.nanmax(X_train[:,11]) - np.nanmax(X_train[:,12]), np.nanmax(X_val[:,11] - X_val[:,12]), np.nanmax(X_test[:,11] - X_test[:,12])))
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = max_features[i]
                elif i==5:
                    max_features[i] = math.ceil(max(np.nanmax(X_train[:,11]) - np.nanmax(X_train[:,13]), np.nanmax(X_val[:,11] - X_val[:,13]), np.nanmax(X_test[:,11] - X_test[:,13])))
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = max_features[i]
                else:
                    max_features[i] = math.ceil(max(np.nanmax(X_train[:,i]), np.nanmax(X_val[:,i]), np.nanmax(X_test[:,i])))
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = max_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = max_features[i]

        else:

            for i in range(nbr_features):

                max_features[i] = np.nanmax(X_train[:,i])
                nan_mask = np.isnan(X_train[:,i])
                nan_idx = np.where(nan_mask)[0]
                X_train_filled[nan_idx,i] = max_features[i]
                nan_mask = np.isnan(X_val[:,i])
                nan_idx = np.where(nan_mask)[0]
                X_val_filled[nan_idx,i] = max_features[i]
                nan_mask = np.isnan(X_test[:,i])
                nan_idx = np.where(nan_mask)[0]
                X_test_filled[nan_idx,i] = max_features[i]

        data_imputation_values = max_features

    elif data_imputation == 'mean':

        mean_features = np.zeros(nbr_features)

        if 'color' in constraints:

            for i in range((nbr_features-1),-1,-1):
                if i ==0:
                    mean_features[i] = np.nanmean(X_train[:,6]) - np.nanmean(X_train[:,7])
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = mean_features[i]
                elif i ==1:
                    mean_features[i] = np.nanmean(X_train[:,7]) - np.nanmean(X_train[:,8])
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = mean_features[i]
                elif i==2:
                    mean_features[i] = np.nanmean(X_train[:,8]) - np.nanmean(X_train[:,9])
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = mean_features[i]
                elif i==3:
                    mean_features[i] = np.nanmean(X_train[:,9]) - np.nanmean(X_train[:,10])
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = mean_features[i]
                elif i==4:
                    mean_features[i] = np.nanmean(X_train[:,11]) - np.nanmean(X_train[:,12])
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = mean_features[i]
                elif i==5:
                    mean_features[i] = np.nanmean(X_train[:,11]) - np.nanmean(X_train[:,13])
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = mean_features[i]
                else:
                    mean_features[i] = np.nanmean(X_train[:,i])
                    nan_mask = np.isnan(X_train[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_train_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_val[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_val_filled[nan_idx,i] = mean_features[i]
                    nan_mask = np.isnan(X_test[:,i])
                    nan_idx = np.where(nan_mask)[0]
                    X_test_filled[nan_idx,i] = mean_features[i]

        else:

            for i in range(nbr_features):

                mean_features[i] = np.nanmean(X_train[:,i])
                nan_mask = np.isnan(X_train[:,i])
                nan_idx = np.where(nan_mask)[0]
                X_train_filled[nan_idx,i] = mean_features[i]
                nan_mask = np.isnan(X_val[:,i])
                nan_idx = np.where(nan_mask)[0]
                X_val_filled[nan_idx,i] = mean_features[i]
                nan_mask = np.isnan(X_test[:,i])
                nan_idx = np.where(nan_mask)[0]
                X_test_filled[nan_idx,i] = mean_features[i]

        data_imputation_values = mean_features

    else:

        for i in range(nbr_features):
            nan_mask = np.isnan(X_train[:,i])
            idx = np.where(nan_mask)[0]
            X_train_filled[idx,i] = data_imputation
            nan_mask = np.isnan(X_val[:,i])
            idx = np.where(nan_mask)[0]
            X_val_filled[idx,i] = data_imputation
            nan_mask = np.isnan(X_test[:,i])
            idx = np.where(nan_mask)[0]
            X_test_filled[idx,i] = data_imputation

        data_imputation_values = np.full(nbr_features, data_imputation)

    return X_train_filled, X_val_filled, X_test_filled, data_imputation_values

def process_sample_weights(sample_weights_train, sample_weights_val, sample_weights_test, zphot_conf_threshold):

    min_conf = min(np.amin(sample_weights_train),np.amin(sample_weights_val),np.amin(sample_weights_test))
    interpolator = interp1d([np.amin(min_conf),1.0],[0.0,1.0])

    for idx, i in enumerate(sample_weights_train):
        sample_weights_train[idx] = interpolator(i)
    for idx, i in enumerate(sample_weights_val):
        sample_weights_val[idx] = interpolator(i)
    for idx, i in enumerate(sample_weights_test):
        sample_weights_test[idx] = interpolator(i)

    return sample_weights_train, sample_weights_val, sample_weights_test

def save_training_data(X_train, data_keys_train, model_path):

    fits_column_train = []
    nbr_features_train = X_train.shape[1]
    for g in range(nbr_features_train):
        fits_column_train.append(fits.Column(name=data_keys_train[g], array=X_train[:,g], format='E'))
    training_data = fits.BinTableHDU.from_columns(fits_column_train)
    try:
        training_data.writeto(os.path.join(model_path, 'training_data.fits'))
    except OSError as e:
        print(e)

    return
