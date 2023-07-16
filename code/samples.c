#include "widereach.h"
#include "helper.h"
#include <string.h>


/* -------------------- Samples ------------------------------------------ */

int is_binary(const samples_t *samples) {
	return 	 2 == samples->class_cnt &&
		-1 == samples->label[0] &&
		 1 == samples->label[1];
}


size_t samples_total(const samples_t *samples) {
	size_t *count = samples->count;
	int cnt = 0;
	for (size_t class = 0; class < samples->class_cnt; class++) {
		cnt += count[class];
	}
	return cnt;
}


size_t positives(const samples_t *samples) {
	return samples->count[1];
}

size_t negatives(const samples_t *samples) {
	return samples->count[0];
}

samples_t *delete_samples(samples_t *samples) {
	free(samples->label);
	size_t class_cnt = samples->class_cnt;
	for (size_t i = 0; i < class_cnt; i++) {
		size_t cnt = samples->count[i];
		for (size_t j = 0; j < cnt; j++) {
			free(samples->samples[i][j]);
		}
		free(samples->samples[i]);
	}
	free(samples->samples);
	free(samples->count);
	return samples;
}

double primary_value(int label) {
    return label > 0 ? 1. : 0.;
}

double **random_points(size_t count, size_t dimension) {
	double **samples = CALLOC(count, double *);
	for (size_t j = 0; j < count; j++) {
		samples[j] = random_point(dimension);
	}
	return samples;
}


void set_sample_class(
		samples_t *samples, 
		size_t class, 
		int label, 
		size_t count) {
	samples->label[class] = label;
	samples->count[class] = count;
	samples->samples[class] = random_points(count, samples->dimension);
}


samples_t *random_samples(
		size_t count, size_t positives, size_t dimension) {
	samples_t *samples = CALLOC(1, samples_t);
	samples->dimension = dimension;
	samples->class_cnt = 2;
	samples->label = CALLOC(2, int);
	samples->count = CALLOC(2, size_t);
	samples->samples = CALLOC(2, double **);
	if (positives > count) {
		positives = count;
	}
	set_sample_class(samples, 0, -1, count - positives);
	set_sample_class(samples, 1,  1, positives);
	return samples;
}


void print_sample(sample_locator_t loc, samples_t *samples) {
	size_t class = loc.class;
	glp_printf("%i, ", samples->label[class]);

	double *sample = samples->samples[class][loc.index];
	size_t dimension = samples->dimension;
	for (size_t j = 0; j < dimension; j++) {
		glp_printf("%g, ", sample[j]);
	}
	glp_printf("\n");

}

void print_samples(samples_t *samples) {
	size_t class_cnt = samples->class_cnt;
	size_t *counts = samples->count;
	for (size_t class = 0; class < class_cnt; class ++) {
		size_t count = counts[class];
		for (size_t i = 0; i < count; i++) {
			sample_locator_t loc = { class, i };
			print_sample(loc, samples);
		}
	}
}


double distance(
        sample_locator_t *loc, 
        samples_t *samples, 
        double *hyperplane, 
		double precision) {
	size_t class = loc->class;
	double *sample = samples->samples[class][loc->index];
    size_t dimension = samples->dimension;
	double product = - hyperplane[dimension] 
                     - samples->label[class] * precision;
	for (size_t i = 0; i < dimension; i++) {
		product += sample[i] * hyperplane[i];
	}
    return product;
}

double sample_violation(
        sample_locator_t *loc, 
        samples_t *samples, 
        double *hyperplane, 
		double precision) {
    double dist = distance(loc, samples, hyperplane, precision);
    if (loc->class > 0) {
        dist = -dist;
    }
    return dist;
}


int side(
		sample_locator_t *loc, 
		samples_t *samples, 
		double *hyperplane, 
		double precision) {
    double dist = distance(loc, samples, hyperplane, precision);
    return 0. == dist ? loc->class : dist > 0.;
}

int side_cnt(
		int class, 
		samples_t *samples, 
		double *hyperplane,
		double precision) {
    sample_locator_t loc;
    loc.class = (int) class;
    int positive_cnt = 0;
    size_t count = samples->count[class];
    for (size_t i = 0; i < count; i++) {
        loc.index = i;
        positive_cnt += side(&loc, samples, hyperplane, precision);
    }
    return positive_cnt;
}

int reduce(
    samples_t *samples,
    void *initial,
    int (*accumulator)(samples_t *, sample_locator_t, void *, void *),
    void *aux) {
  void *result = initial;
  size_t classes[] = { 1, 0 };
  for (int class_index = 0; class_index < 2; class_index++) {
    size_t class = classes[class_index];
    int cnt = samples->count[class];
    for (size_t idx = 0; idx < cnt; idx++) {
      sample_locator_t locator = { class, idx };
      int state = accumulator(samples, locator, result, aux);
      if (state != 0) {
        return state;
      }
	}
  }

  return 0;
}


void add_bias(samples_t *samples) {
  for(int class = 0; class < 2; class++) {
    for(size_t i = 0; i < samples->count[class]; i++) {
      double *s = samples->samples[class][i];
      //double *new_s = realloc(s, samples->dimension+1);
      double *new_s = CALLOC(samples->dimension+1, double);
      memcpy(new_s, s, sizeof(double)*samples->dimension);
      if(!new_s) {
	printf("add_bias: realloc failed\n");
	exit(EXIT_FAILURE);
      }
      s = new_s;
      s[samples->dimension] = 1;
      samples->samples[class][i] = s;
    }
  }
  samples->dimension++;
}
