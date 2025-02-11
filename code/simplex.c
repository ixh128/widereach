#include "widereach.h"
#include "helper.h"
#include <math.h>

void mirror_sample(size_t dimension, double *sample) {
  for (size_t i = 0; i < dimension; i++) {
    sample[i] = 1. - sample[i];
  }
}

double exp_noise(double scale) {
  //random sample from exponential distribution with the given scale
  if(scale == 0)
    return 0;
  double rate = 1/scale;
  double unif = drand48();
  return (-1/rate)*log(1-unif); //inverse of exponential cdf
}

void fix_bound(size_t dimension, double *sample) {
  //for each dimension, if the sample lies outside of the hypercube, we uniformly distribute it inside of it instead
  for(size_t i = 0; i < dimension; i++) {
    if(sample[i] < 0 || sample[i] > 1) {
      sample[i] = drand48();
    }
  }
}

double **random_simplex_points_old(size_t count, simplex_info_t *simplex_info) {
  /* this is the old function to generate the cluster benchmark */
  double **samples = CALLOC(count, double *);
  size_t count_simplex = count * (1 - simplex_info->noise_ratio);
  size_t mirror_count = count_simplex / simplex_info->cluster_cnt;
  size_t dimension = simplex_info->dimension;
  for (size_t j = 0; j < count_simplex; j++) {
    double side = simplex_info->side + exp_noise(simplex_info->scale);
    samples[j] = random_simplex_point(side, dimension);
    fix_bound(dimension, samples[j]); //ensures it is inside the hypercube (noise can affect that)
    if (j >= mirror_count) {
      mirror_sample(dimension, samples[j]);
    }
  }
  for (size_t j = count_simplex; j < count; j++) {
    samples[j] = random_point(dimension);
  }
  return samples;
}

void shuffle(size_t *arr, size_t n) {
  /** rearrange elements of arr in a random order */
  if (n > 1) {
    for (size_t i = 0; i < n - 1; i++) {
      size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
      int tmp = arr[j];
      arr[j] = arr[i];
      arr[i] = tmp;
    }
  }
}

double **random_simplex_points(size_t count, simplex_info_t *simplex_info) {
  double **samples = CALLOC(count, double *);
  size_t count_simplex = count / 2;
  size_t *cutoff_pts = CALLOC(simplex_info->cluster_cnt, size_t); //the indices at which we switch from one cluster to the next
  if(simplex_info->cluster_sizes == NULL) {
    size_t cluster_size = count_simplex / simplex_info->cluster_cnt;
    cutoff_pts[0] = cluster_size;
    for(int i = 1; i < simplex_info->cluster_cnt; i++) {
      cutoff_pts[i] = cutoff_pts[i-1] + cluster_size;
    }
  } else {
    cutoff_pts[0] = simplex_info->cluster_sizes[0];
    for(int i = 1; i < simplex_info->cluster_cnt; i++) {
      cutoff_pts[i] = cutoff_pts[i-1] + simplex_info->cluster_sizes[i];
    }
  }
  size_t dimension = simplex_info->dimension;

  size_t *cluster_order = CALLOC(simplex_info->dimension, size_t);
  for(int i = 0; i < simplex_info->dimension; i++) {
    cluster_order[i] = i;
  }
  shuffle(cluster_order, simplex_info->dimension); //shuffle the whole array, but we may not use all of it
  /*for(int i = 0; i < simplex_info->cluster_cnt; i++)
    printf("%s%ld", i == 0 ? "cluster order: " : ", ", cluster_order[i]);
    printf("\n");*/


  size_t cluster_idx = 0; //which cluster is currently being added to
  for (size_t j = 0; j < count_simplex; j++) {
    samples[j] = random_simplex_point(simplex_info->side, dimension);
    /*if (j >= cluster_size) {
      mirror_sample(dimension, samples[j]);
      }*/
    if(j >= cutoff_pts[cluster_idx]) {
      cluster_idx++;
    }
    if(simplex_info->cluster_cnt == 2 && j >= cutoff_pts[0]) { //special case for 2-cluster: diagonally opposite clusters
      mirror_sample(dimension, samples[j]);
    } else { //otherwise, proceed assuming that we fill all corners sequentially with clusters
      for(size_t i = 0; i < dimension; i++) {
	if(cluster_order[cluster_idx] & (1 << i)) { //if the ith bit in the cluster idx is set, we should mirror over the ith dimension
	  samples[j][i] = 1 - samples[j][i];
	}
      }
    }
  }
  for (size_t j = count_simplex; j < count; j++) {
    samples[j] = random_point(dimension);
  }
  free(cluster_order);
  free(cutoff_pts);
  return samples;
}

void set_sample_class_simplex(
		samples_t *samples, 
		size_t class, 
		int label, 
        size_t count,
		simplex_info_t *simplex_info) {
	samples->label[class] = label;
	samples->count[class] = count;
	samples->samples[class] = random_simplex_points_old(count, simplex_info);
}

samples_t *random_simplex_samples(simplex_info_t *simplex_info) {
  samples_t *samples = CALLOC(1, samples_t);
  samples->dimension = simplex_info->dimension;
  samples->class_cnt = 2;
  samples->label = CALLOC(2, int);
  samples->count = CALLOC(2, size_t);
  samples->samples = CALLOC(2, double **);
  size_t *positives = &(simplex_info->positives);
  size_t count = simplex_info->count;
  if (*positives > count) {
    *positives = count;
  }
  set_sample_class(samples, 0, -1, count - *positives);
  set_sample_class_simplex(samples, 1,  1, *positives, simplex_info);
  return samples;
}

double **random_prism_points(size_t count, simplex_info_t *simplex_info, size_t dims) {
  //generates the simplex in "dims" dimensions. remaining dimensions are uniform
  double **samples = CALLOC(count, double *);
  size_t count_simplex = count * (1 - simplex_info->noise_ratio);
  size_t mirror_count = count_simplex / simplex_info->cluster_cnt;
  size_t dimension = simplex_info->dimension;
  for (size_t j = 0; j < count_simplex; j++) {
    double side = simplex_info->side + exp_noise(simplex_info->scale);
    samples[j] = random_prism_point(side, dimension, dims);
    fix_bound(dimension, samples[j]); //ensures it is inside the hypercube (noise can affect that)
    if (j >= mirror_count) {
      mirror_sample(dims, samples[j]);
    }
  }
  for (size_t j = count_simplex; j < count; j++) {
    samples[j] = random_point(dimension);
  }
  return samples;
}

double **random_prism_points_unbalanced(size_t count, simplex_info_t *simplex_info, size_t dims, double ratio, double side1, double side2) {
  //generates the simplex in "dims" dimensions. remaining dimensions are uniform
  //ratio = fraction of cluster points which should be in the first cluster
  //side1 and 2 are side lengths of those clusters
  double **samples = CALLOC(count, double *);
  size_t count_simplex = count * (1 - simplex_info->noise_ratio);
  //size_t mirror_count = count_simplex / simplex_info->cluster_cnt;
  size_t mirror_count = ratio * count_simplex; //just to test imbalanced clusters
  size_t dimension = simplex_info->dimension;
  for (size_t j = 0; j < count_simplex; j++) {
    if (j >= mirror_count) {
      double side = side2 + exp_noise(simplex_info->scale);
      samples[j] = random_prism_point(side, dimension, dims);
      fix_bound(dimension, samples[j]); //ensures it is inside the hypercube (noise can affect that)
      mirror_sample(dims, samples[j]);
    } else {
      double side = side1 + exp_noise(simplex_info->scale);
      samples[j] = random_prism_point(side, dimension, dims);
      fix_bound(dimension, samples[j]);
    }
  }
  for (size_t j = count_simplex; j < count; j++) {
    samples[j] = random_point(dimension);
  }
  return samples;
}

void set_sample_class_prism(
			    samples_t *samples, 
			    size_t class, 
			    int label, 
			    size_t count,
			    simplex_info_t *simplex_info,
			    size_t simplex_dims) {
  samples->label[class] = label;
  samples->count[class] = count;
  samples->samples[class] = random_prism_points(count, simplex_info, simplex_dims);
}


samples_t *random_prism_samples(simplex_info_t *simplex_info, size_t simplex_dims) {
  samples_t *samples = CALLOC(1, samples_t);
  samples->dimension = simplex_info->dimension;
  samples->class_cnt = 2;
  samples->label = CALLOC(2, int);
  samples->count = CALLOC(2, size_t);
  samples->samples = CALLOC(2, double **);
  size_t *positives = &(simplex_info->positives);
  size_t count = simplex_info->count;
  if (*positives > count) {
    *positives = count;
  }
  set_sample_class(samples, 0, -1, count - *positives);
  set_sample_class_prism(samples, 1,  1, *positives, simplex_info, simplex_dims);
  return samples;
}

double *prism_cut_hplane(simplex_info_t *simplex_info, size_t simplex_dims) {
  double *w = CALLOC(simplex_info->dimension + 1, double);
  for(int i = 0; i < simplex_dims; i++) {
    w[i] = -1;
  }
  w[simplex_info->dimension] = -simplex_info->side;
  return w;
}

