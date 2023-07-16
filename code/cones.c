#include "widereach.h"
#include "helper.h"

#include <math.h>
#include <string.h>
#include <gsl/gsl_blas.h>

int print_mat(FILE *f, const gsl_matrix *m) {
  int status, n = 0;

  for (size_t i = 0; i < m->size1; i++) {
    for (size_t j = 0; j < m->size2; j++) {
      if ((status = fprintf(f, "%g ", gsl_matrix_get(m, i, j))) < 0)
        return -1;
      n += status;
    }

    if ((status = fprintf(f, "\n")) < 0)
      return -1;
    n += status;
  }

  return n;
}

int is_zero_vec(double *x, size_t dimension) {
  int out = 1;
  for(size_t i = 0; i < dimension; i++) {
    if(x[i] != 0)
      out = 0;
  }
  return out;
}

double *apply_proj(gsl_matrix *c, double *x, size_t dimension) {
  gsl_vector u = gsl_vector_view_array(x, dimension).vector;
  gsl_vector *res = gsl_vector_alloc(dimension);
  gsl_blas_dgemv(CblasNoTrans, 1, c, &u, 0, res);
  double *y = CALLOC(dimension, double);
  memcpy(y, res->data, dimension*sizeof(double));
  gsl_vector_free(res);
  /*if(is_zero_vec(y, dimension))
    printf("y = 0\n");*/
  return y;
}

gsl_matrix *add_new_proj(gsl_matrix *c, double *uA) {
  //compute C' = C - uuT/uTu
  gsl_vector u = gsl_vector_view_array(uA, c->size2).vector;
  double denom;
  gsl_blas_ddot(&u, &u, &denom);
  gsl_matrix *new_c = gsl_matrix_alloc(c->size1, c->size2);
  gsl_matrix_memcpy(new_c, c);
  gsl_blas_dger(-1/denom, &u, &u, new_c);
  return new_c;
}

sparse_vector_t *copy_sparse_vector(sparse_vector_t *v) {
  //sparse_vector_t *cp = sparse_vector_blank(v->len + v->extra);
  sparse_vector_t *cp = CALLOC(1, sparse_vector_t);
  cp->len = v->len;
  cp->extra = v->extra;
  cp->ind = CALLOC(v->len+v->extra+1, int);
  cp->val = CALLOC(v->len+v->extra+1, double);
  memcpy(cp->ind, v->ind, (v->len+v->extra+1)*sizeof(int));
  memcpy(cp->val, v->val, (v->len+v->extra+1)*sizeof(double));
  return cp;
}

double compute_obj(env_t *env, double *w, gsl_matrix *c) {
  size_t reach = 0, nfpos = 0;
  size_t dimension = env->samples->dimension;
  for(size_t class = 0; class < env->samples->class_cnt; class++) {
    for(size_t i = 0; i < env->samples->count[class]; i++) {
      double *cs = apply_proj(c, env->samples->samples[class][i], dimension);
      double dot = 0;
      for(size_t k = 0; k < dimension; k++) {
	dot += w[k]*cs[k];
      }
      if(dot >= 1e-15) { //TODO: this threshold might be bad
	if(class == 1) {
	  /*printf("true pos: ");
	  for(size_t k = 0; k < dimension; k++) {
	    printf("%g%s", cs[k], k == dimension-1 ? "\n" : ", ");
	    }*/
	  reach++;
	}
	else {
	  /*printf("false pos: ");
	  for(size_t k = 0; k < dimension; k++) {
	    printf("%g%s", cs[k], k == dimension-1 ? "\n" : ", ");
	    }*/
	  nfpos++;
	}
      }
      free(cs);
    }
  }
  double prec;
  if(reach + nfpos == 0)
    prec = 0;
  else
    prec = ((double) reach)/(reach + nfpos);
  //printf("reach = %ld, # false pos = %ld, prec = %g\n", reach, nfpos, prec);
  if(prec < env->params->theta)
    return -1;
  else
    return reach;
}

double *solve_exact_closed(env_t *env, size_t D, sparse_vector_t *V, gsl_matrix *c) {
  //printf("Running, D = %lu\n", D);
  size_t dimension = env->samples->dimension;
  int all_zero = 1;
  for(size_t class = 0; class < env->samples->class_cnt; class++) {
    for(size_t i = 0; i < env->samples->count[class]; i++) {
      double *cs = apply_proj(c, env->samples->samples[class][i], dimension);
      if(!is_zero_vec(cs, dimension)) {
	all_zero = 0;
	free(cs);
	break;
      }
      free(cs);
    }
    if(!all_zero) break;
  }
  if(all_zero) {
    printf("all zero\n");
    return CALLOC(dimension, double);
  }
  if(D == 1) {
    double *u1 = CALLOC(dimension, double);
    double *u2 = CALLOC(dimension, double);
    for(size_t i = 0; i < dimension; i++) {
      u1[i] = 1;
      u2[i] = -1;
    }
    double *v1 = apply_proj(c, u1, dimension);
    double *v2 = apply_proj(c, u2, dimension);
    free(u1);
    free(u2);
    double obj1 = compute_obj(env, v1, c);
    double obj2 = compute_obj(env, v2, c);
    //printf("obj1 = %g, obj2 = %g\n", obj1, obj2);
    if(obj1 > obj2) {
      free(v2);
      return v1;
    } else {
      free(v1);
      return v2;
    }
  }
  printf("done with base cases\n");

  double best_obj = -1e101;
  double *best_w = NULL;

  for(size_t class = 0; class < env->samples->class_cnt; class++) {
    for(size_t i=0; i < env->samples->count[class]; i++) {
      printf("about to project\n");
      double *u = apply_proj(c, env->samples->samples[class][i], dimension);
      printf("projected\n");
      if(is_zero_vec(u, dimension)) {
	free(u);
	continue;
      }
      sparse_vector_t *newV = copy_sparse_vector(V);
      append(newV, class, i); //probably not a good idea to use sparse_vector for this; instead, use a regular vector and just store a pair
      gsl_matrix *newc = add_new_proj(c, u);
      printf("constructed newc\n");
      /*printf("u = ");
      for(size_t i = 0; i < dimension; i++)
	printf("%g%s", u[i], i == dimension - 1 ? "\n" : ", ");
      printf("c = \n");
      print_mat(stdout, newc);*/
      printf("entering recursion...\n");
      double *w = solve_exact_closed(env, D-1, newV, newc);
      printf("finished recursion\n");
      double obj = compute_obj(env, w, c); //this could be eliminated by returning the objective from this function as well as the hyperplane
      printf("computed obj\n");
      /*printf("D = %ld, w = ", D);
      for(size_t i = 0; i < dimension; i++)
	printf("%g%s", w[i], i == dimension - 1 ? "\n" : ", ");
	printf(" => obj = %g\n", obj);*/
      if(obj > best_obj) {
	best_obj = obj;
	if(best_w != NULL)
	  free(best_w);
	best_w = w;
      } else {
	free(w);
      }
      gsl_matrix_free(newc);
      free(u);
    }
  }

  printf("Stepping out. D = %lu, obj = %g\n", D, best_obj);
    printf(" -> best_w = ");
  for(size_t i = 0; i < dimension; i++)
    printf("%g%s", best_w[i], i == dimension - 1 ? "\n" : ", ");
  return best_w;
}

double *single_exact_run(env_t *env) {
  size_t dimension = env->samples->dimension;
  sparse_vector_t *V = sparse_vector_blank(dimension+1);
  gsl_matrix *c = gsl_matrix_alloc(dimension, dimension);
  for(size_t i = 0; i < dimension; i++)
    gsl_matrix_set(c, i, i, 1);
  double *w = solve_exact_closed(env, env->samples->dimension, V, c);
  printf("obj = %g\n", compute_obj(env, w, c));
  return w;
}
