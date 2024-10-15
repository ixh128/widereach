#include <math.h>
#include <float.h>

#include "widereach.h"
#include "helper.h"

/*
 * Translates indexing in a problem instance (env) into GLPK problem (glp_prob).
 *
 * Numbering Conventions
 *
 * Variables (columns)
 * 1 to dimension w
 * dimension+1c
 * dimension+2 to dimension+positives+1 xi
 * dimension+positives+2 to dimension+samples+1	yj
 * dimension+samples+2V
 *
 * Constrains (rows)
 * 1 to positives xi
 * positives+1 to samples yj
 * samples+1V 
 */

int directional_offset(int index, int direction, size_t dimension) {
	int index_shifted = index;
	if (!direction) {
		index_shifted += dimension + 1;
	}
	return index_shifted;
}


int idx(int direction, int class, size_t sample_index, samples_t *samples) { 
	int idx = (int) sample_index + 1;
	idx = directional_offset(idx, direction, samples->dimension);
	if (samples->label[class] < 0) {
		idx += positives(samples);
	}
	return idx;
}


int violation_idx(int direction, samples_t *samples) {
	int idx = samples_total(samples) + 1;
	idx = directional_offset(idx, direction, samples->dimension);
	return idx;
}


sample_locator_t *locator(int index, samples_t *samples) {
	sample_locator_t *locator = CALLOC(1, sample_locator_t);
	int index_min = samples->dimension + 2;
	if (index < index_min) {
		locator->class = -1;
		return locator;
	}
	size_t offset = (size_t) (index - index_min);
	if (offset >= samples_total(samples)) {
		locator->class = -1;
		return locator;
	}

	size_t threshold = positives(samples);
	if (offset >= threshold) {
		locator->class = 0;
		locator->index = offset - threshold;
	} else {
		locator->class = 1;
		locator->index = offset;
	}

	return locator;
}

int append_locator(
        sparse_vector_t *v, 
        sample_locator_t *loc, 
        double val, 
        samples_t *samples) {
    return append(v, idx(0, loc->class, loc->index, samples), val);
}

int idx_extreme(int direction, int class, int extreme, samples_t *samples) {
	int label = samples->label[class];
	int idx;
	if (label > 0) {
		idx = !extreme ? 1: positives(samples);
	} else {
		idx = !extreme ? 
			positives(samples) + 1 : samples_total(samples);
	}
	idx = directional_offset(idx, direction, samples->dimension);
	return idx;
}


double label_to_penalty(int label, double theta) {
	return label > 0 ? theta - 1. : theta;
}

int idx_thr(int direction, int class, size_t sample_index, samples_t *samples) { 
	int idx = (int) sample_index + 1;
	idx = directional_offset(idx, direction, samples->dimension);
	idx += positives(samples) + negatives(samples);
	if (samples->label[class] < 0) {
		idx += positives(samples);
	}
	return idx;
}

sparse_vector_t *cover_row_thr(
		sparse_vector_t *constraint, 
		size_t class, 
		double coef, 
		samples_t *samples) {
	int count = samples->count[class];
	for (size_t i = 0; i < count; i++) { 
		append(constraint, idx_thr(0, class, i, samples), coef);
	}
	return constraint;
}

sparse_vector_t *precision_row_thr(samples_t *samples, double theta) {
	size_t len = samples_total(samples) + 2;
	sparse_vector_t *constraint = sparse_vector_blank(len);
	size_t class_cnt = samples->class_cnt;
	for (size_t class = 0; class < class_cnt; class++) {
		double penalty = label_to_penalty(samples->label[class], theta);
		cover_row_thr(constraint, class, penalty, samples);
	}
	append(constraint, violation_idx(0, samples), -1.);
	return constraint;
}

sparse_vector_t *precision_row(samples_t *samples, double theta) {
	size_t len = samples_total(samples) + 2;
	sparse_vector_t *constraint = sparse_vector_blank(len);
	size_t class_cnt = samples->class_cnt;
	for (size_t class = 0; class < class_cnt; class++) {
		double penalty = label_to_penalty(samples->label[class], theta);
		cover_row(constraint, class, penalty, samples);
	}
	append(constraint, violation_idx(0, samples), -1.);
	return constraint;
}

sparse_vector_t *precision_row_strict(samples_t *samples, double theta) {
	size_t len = samples_total(samples) + 2;
	sparse_vector_t *constraint = sparse_vector_blank(len);
	size_t class_cnt = samples->class_cnt;
	for (size_t class = 0; class < class_cnt; class++) {
		double penalty = label_to_penalty(samples->label[class], theta);
		cover_row(constraint, class, penalty, samples);
	}
	//append(constraint, violation_idx(0, samples), -1.);
	return constraint;
}

sparse_vector_t *cover_row(
		sparse_vector_t *constraint, 
		size_t class, 
		double coef, 
		samples_t *samples) {
	int count = samples->count[class];
	for (size_t i = 0; i < count; i++) { 
		append(constraint, idx(0, class, i, samples), coef);
	}
	return constraint;
}

int index_to_class(int idx, samples_t *samples) {
	sample_locator_t *loc = locator(idx, samples);
	int class = loc->class;
	free(loc);
	return class;
}

int index_label(int i, samples_t *samples) {
	int class = index_to_class(i, samples);
	glp_assert(class >= 0);
	return samples->label[class];
}

double *blank_solution(samples_t *samples) {
    int idx_max = violation_idx(0, samples);
    double *solution = CALLOC(idx_max + 1, double);
    for (int i = 1; i <= idx_max; i++) {
        solution[i] = .5;
    }
    return solution;
}

void update_solution_element(double *solution, int index, double value) {
    if (solution != NULL && solution[index] != ceil(solution[index])) {
        solution[index] = value;
    }
}

double hinge(int cond, double v) {
    return cond ? v : 0.;
}

void hyperplane_to_distance(
        double *hyperplane, 
        double *dist, 
        env_t *env) {
    if (NULL == hyperplane) {
        return;
    }
    
    params_t *params = env->params;
    double precision[] = { params->epsilon_negative, params->epsilon_positive };
    samples_t *samples = env->samples;
    int idx_min = idx_extreme(0, 1, 0, samples);
    int idx_max = violation_idx(0, samples);
    for (int i = idx_min; i < idx_max; i++) {
        sample_locator_t *loc = locator(i, samples);
        dist[i] = distance(loc, samples, hyperplane, precision[loc->class]);
        free(loc);
    }
}

unsigned int count_solution(double *solution, int count) {
  unsigned int result = 0;
  for (int i = 0; i < count; i++) {
    result += (int) solution[i];
  }
  return result;
}

unsigned int reach(double *solution, samples_t *samples) {
  return count_solution(solution + idx_extreme(0, 1, 0, samples), 
                        positives(samples));
}

unsigned int false_positives(double *solution, samples_t *samples) {
  return count_solution(solution + idx_extreme(0, 0, 0, samples),
                        negatives(samples));
}


double precision(double *solution, samples_t *samples) {
  double r = (double) reach(solution, samples);
  double fp = (double) false_positives(solution, samples);
  return r / (r + fp);
}

double hyperplane_to_solution_parts(
        double *hyperplane, 
        double *solution,
        params_t *params,
        samples_t *samples) {
  if (NULL == hyperplane) {
    return -DBL_MAX;
  }
    
  double theta = params->theta;
  double violation = theta * params->epsilon_precision;
  double violation_coefficient[] = { theta, theta - 1. };
  double precision[] = { params->epsilon_negative, params->epsilon_positive };
  int idx_min = idx_extreme(0, 1, 0, samples);
  int idx_max = violation_idx(0, samples);
  double value = 0.;
  for (int i = idx_min; i < idx_max; i++) {
    sample_locator_t *loc = locator(i, samples);
    int class = loc->class;
    double v = side(loc, samples, hyperplane, precision[class]);
    free(loc);
    update_solution_element(solution, i, v);
    violation += v * violation_coefficient[class];
    value += hinge(class, v);
  } 
  violation = hinge(violation >= 0., violation);
  if (solution != NULL) {
    copy_hyperplane(samples->dimension, solution + 1, hyperplane);
    solution[idx_max] = violation;
  }
  value -= params->lambda * violation;
  
  return value;
}

//remove this:
double hyperplane_to_solution_parts_print(
        double *hyperplane, 
        double *solution,
        params_t *params,
        samples_t *samples) {
  if (NULL == hyperplane) {
    return -DBL_MAX;
  }
    
  double theta = params->theta;
  double violation = theta * params->epsilon_precision;
  double violation_coefficient[] = { theta, theta - 1. };
  double precision[] = { params->epsilon_negative, params->epsilon_positive };
  int idx_min = idx_extreme(0, 1, 0, samples);
  int idx_max = violation_idx(0, samples);
  double value = 0.;
  for (int i = idx_min; i < idx_max; i++) {
    sample_locator_t *loc = locator(i, samples);
    int class = loc->class;
    double v = side(loc, samples, hyperplane, precision[class]);
    //free(loc);
    update_solution_element(solution, i, v);
    violation += v * violation_coefficient[class];
    value += hinge(class, v);
    printf("sample (%d, %ld): ", class, loc->index);
    for(int i = 0; i < samples->dimension; i++) {
      printf("%0.3f ", samples->samples[class][loc->index][i]);
    }
    printf("\n => v = %g\n", v);
    free(loc);
  } 
  violation = hinge(violation >= 0., violation);
  if (solution != NULL) {
    copy_hyperplane(samples->dimension, solution + 1, hyperplane);
    solution[idx_max] = violation;
  }
  value -= params->lambda * violation;
  
  return value;
}

double hyperplane_to_solution(
        double *hyperplane, 
        double *solution,
        env_t *env) {
  return hyperplane_to_solution_parts(hyperplane, 
                                      solution, 
                                      env->params, 
                                      env->samples);
}


struct distance_record {
    int index;
    double directional_distance;
};

int dist_cmp(const void *a, const void *b) {
    struct distance_record *r1 = (struct distance_record *) a;
    struct distance_record *r2 = (struct distance_record *) b;
    double d1 = r1->directional_distance;
    double d2 = r2->directional_distance;
    if (d1 > d2) {
        return 1;
    } else if (d1 < d2) {
        return -1;
    } else {
        return 0;
    }
}

int *sorted_by_violation(double *hyperplane, env_t *env) {
    if (NULL == hyperplane) {
        return NULL;
    }
    
    params_t *params = env->params;
    double precision[] = { params->epsilon_negative, params->epsilon_positive };
    samples_t *samples = env->samples;
    size_t dimension = violation_idx(0, samples);
    struct distance_record *dist = CALLOC(dimension, struct distance_record);
    int idx_min = idx_extreme(0, 1, 0, samples);
    for (int i = idx_min; i < dimension; i++) {
        dist[i].index = i;
        sample_locator_t *loc = locator(i, samples);
        int class = loc->class;
        double dist_dir = 
            // sample_violation(loc, samples, hyperplane, precision[class]);
            fabs(distance(loc, samples, hyperplane, precision[class]));
        free(loc);
        if (dist_dir < 0.) {
            dist_dir = DBL_MAX;
        }
        dist[i].directional_distance = dist_dir;
    }
    size_t ary_size = dimension - idx_min;
    qsort(dist + idx_min, ary_size, sizeof(struct distance_record), dist_cmp);
    int *sorted_index = CALLOC(ary_size, int);
    for (int i = 0; i < ary_size; i++) {
        int idx = idx_min + i;
        /* glp_printf("violation(%i)=%g\n", 
                   dist[idx].index, dist[idx].directional_distance); */
        sorted_index[i] = 
            dist[idx].directional_distance < DBL_MAX ? dist[idx].index : 0;
    }
    free(dist);
    return sorted_index;
}
