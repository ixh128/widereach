#include "widereach.h"
#include "helper.h"

#include <math.h>
#include <memory.h>
#include <time.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_linalg.h>

#define NUM_THREADS 16

vsamples_t *samples_to_vec(samples_t *samples) {
  size_t dimension = samples->dimension;
  vsamples_t *vs = malloc(sizeof(vsamples_t));
  vs->dimension = dimension;
  vs->class_cnt = samples->class_cnt;
  vs->count = CALLOC(vs->class_cnt, size_t);
  memcpy(vs->count, samples->count, vs->class_cnt*sizeof(size_t));
  vs->label = CALLOC(vs->class_cnt, int);
  memcpy(vs->label, samples->label, vs->class_cnt*sizeof(int));
  vs->samples = CALLOC(vs->class_cnt, gsl_vector **);
  for(size_t class = 0; class < vs->class_cnt; class++) {
    vs->samples[class] = CALLOC(vs->count[class], gsl_vector *);
    for(size_t i = 0; i < vs->count[class]; i++) {
      vs->samples[class][i] = gsl_vector_calloc(dimension);
      *vs->samples[class][i] = gsl_vector_view_array(samples->samples[class][i], dimension).vector;
    }
  }
  return vs;
}

void vsamples_free(vsamples_t *vs) {
  free(vs->label);
  for(size_t i = 0; i < vs->class_cnt; i++) {
    for(size_t j = 0; j < vs->count[i]; j++) {
      gsl_vector_free(vs->samples[i][j]);
    }
    free(vs->samples[i]);
  }
  free(vs->samples);
  free(vs->count);
  free(vs);
}

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

int is_zero_vec(gsl_vector *x) {
  int out = 1;
  for(size_t i = 0; i < x->size; i++) {
    if(gsl_vector_get(x, i) != 0)
      out = 0;
  }
  return out;
}

gsl_vector *apply_proj(const gsl_matrix *c, const gsl_vector *x) {
  gsl_vector *res = gsl_vector_calloc(x->size);
  gsl_blas_dgemv(CblasNoTrans, 1, c, x, 0, res);
  return res;
}

gsl_matrix *add_new_proj(const gsl_matrix *c, const gsl_vector *x) {
  //compute C' = C - uuT/uTu
  double denom;
  gsl_blas_ddot(x, x, &denom);
  gsl_matrix *new_c = gsl_matrix_calloc(c->size1, c->size2);
  gsl_matrix_memcpy(new_c, c);
  gsl_blas_dger(-1/denom, x, x, new_c);
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

obj_result compute_obj(env_t *env, vsamples_t *vs, gsl_vector *w, gsl_matrix *c) {
  size_t reach = 0, nfpos = 0;
  size_t dimension = env->samples->dimension;
  for(size_t class = 0; class < env->samples->class_cnt; class++) {
    for(size_t i = 0; i < env->samples->count[class]; i++) {
      gsl_vector *cs;
      if(c)
	cs = apply_proj(c, vs->samples[class][i]);
      else
	cs = vs->samples[class][i];
      double dot = 0;
      gsl_blas_ddot(w, cs, &dot);
      if(dot >= 1e-15) { //TODO: this threshold might be bad
	if(class == 1) {
	  reach++;
	}
	else {
	  nfpos++;
	}
      }
      if(c)
	gsl_vector_free(cs);
    }
  }
  double prec;
  if(reach + nfpos == 0)
    prec = 0;
  else
    prec = ((double) reach)/(reach + nfpos);
  /*if(prec < env->params->theta)
    return -1;
  else
  return reach;*/
  double violation = (env->params->theta - 1)*reach + env->params->theta*nfpos;
  if(violation < 0)
    violation = 0;
  return (obj_result) {.obj = reach - violation * env->params->lambda, .prec = prec};
}

obj_result compute_obj_mixed(env_t *env, vsamples_t *vs, gsl_vector *w, gsl_matrix *c, int **m) {
  size_t reach = 0, nfpos = 0;
  size_t dimension = env->samples->dimension;
  for(size_t class = 0; class < env->samples->class_cnt; class++) {
    for(size_t i = 0; i < env->samples->count[class]; i++) {
      gsl_vector *cs;
      if(c)
	cs = apply_proj(c, vs->samples[class][i]);
      else
	cs = vs->samples[class][i];
      double dot = 0;
      gsl_blas_ddot(w, cs, &dot);
      if(!m[class][i]) {
	//closed point
	if(dot >= 1e-15) { //TODO: this threshold might be bad
	  if(class == 1) {
	    reach++;
	  } else {
	    nfpos++;
	  }
	}
      } else {
	//open point
	if(dot > 0) {
	  if(class == 1) {
	    reach++;
	  } else {
	    nfpos++;
	  }
	}
      }
      if(c)
	gsl_vector_free(cs);
    }
  }
  double prec;
  if(reach + nfpos == 0)
    prec = 0;
  else
    prec = ((double) reach)/(reach + nfpos);
  /*if(prec < env->params->theta)
    return -1;
  else
  return reach;*/
  double violation = (env->params->theta - 1)*reach + env->params->theta*nfpos;
  if(violation < 0)
    violation = 0;
  return (obj_result) {.obj = reach - violation * env->params->lambda, .prec = prec};
}

gsl_vector *solve_exact_closed(env_t *env, vsamples_t *vs, size_t D, gsl_matrix *c) {
  //printf("Running, D = %lu\n", D);
  size_t dimension = env->samples->dimension;
  int all_zero = 1;
  for(size_t class = 0; class < env->samples->class_cnt; class++) {
    for(size_t i = 0; i < env->samples->count[class]; i++) {
      gsl_vector *cs = apply_proj(c, vs->samples[class][i]);
      if(!is_zero_vec(cs)) {
	all_zero = 0;
	gsl_vector_free(cs);
	break;
      }
      gsl_vector_free(cs);
    }
    if(!all_zero) break;
  }
  if(all_zero) {
    printf("all zero\n");
    return gsl_vector_calloc(dimension);
  }
  if(D == 1) {
    gsl_vector *u1 = gsl_vector_calloc(dimension);
    gsl_vector_set_all(u1, 1);
    gsl_vector *v1 = apply_proj(c, u1);
    gsl_vector *v2 = gsl_vector_calloc(dimension);
    gsl_vector_axpby(-1, v1, 0, v2); 
    gsl_vector_free(u1);
    double obj1 = compute_obj(env, vs, v1, c).obj;
    double obj2 = compute_obj(env, vs, v2, c).obj;
    printf("obj1 = %g, obj2 = %g\n", obj1, obj2);
    if(obj1 > obj2) {
      gsl_vector_free(v2);
      return v1;
    } else {
      gsl_vector_free(v1);
      return v2;
    }
  }

  double best_obj = -1e101;
  gsl_vector *best_w = NULL;

  for(size_t class = 0; class < env->samples->class_cnt; class++) {
    for(size_t i=0; i < env->samples->count[class]; i++) {
      gsl_vector *u = apply_proj(c, vs->samples[class][i]);
      printf("Chose point (%lu, %lu). u = ", class, i);
      for(int k = 0; k < u->size; k++)
	printf("%g%s", gsl_vector_get(u, k), k == u->size-1 ? "\n" : " ");
      if(is_zero_vec(u)) {
	gsl_vector_free(u);
	continue;
      }
      //Not necessary to store V - we can get the basis vectors using c
      /*sparse_vector_t *newV = copy_sparse_vector(V);
	append(newV, class, i); //probably not a good idea to use sparse_vector for this; instead, use a regular vector and just store a pair*/
      gsl_matrix *newc = add_new_proj(c, u);
      gsl_vector *w = solve_exact_closed(env, vs, D-1, newc);
      double obj = compute_obj(env, vs, w, c).obj; //this could be eliminated by returning the objective from this function as well as the hyperplane
      if(obj > best_obj) {
	best_obj = obj;
	if(best_w != NULL)
	  gsl_vector_free(best_w);
	best_w = w;
      } else {
	gsl_vector_free(w);
      }
      if(obj == 2)
	print_matrix(stdout, newc);
      gsl_matrix_free(newc);
      gsl_vector_free(u);
    }
  }

  printf("Stepping out. D = %lu, obj = %g\n", D, best_obj);
  printf(" -> best_w = ");
  for(size_t i = 0; i < dimension; i++)
    printf("%g%s", gsl_vector_get(best_w, i), i == dimension - 1 ? "\n" : ", ");
  return best_w;
}

int **build_new_m(vsamples_t *vs, gsl_matrix *c, gsl_vector *u, int **m) {
  int **new_m = CALLOC(vs->class_cnt, int *);
  for(int k = 0; k < vs->class_cnt; k++) {
    new_m[k] = CALLOC(vs->count[k], int);
    for(int l = 0; l < vs->count[k]; l++) {
      gsl_vector *cs = apply_proj(c, vs->samples[k][l]);
      double dot;
      gsl_blas_ddot(u, cs, &dot);
      gsl_vector_free(cs);
      if((m[k][l] == 0 && dot >= 0) || (m[k][l] == 1 && dot > 0))
	new_m[k][l] = 0;
      else
	new_m[k][l] = 1;
    }
  }
  return new_m;
}

gsl_vector *solve_exact_mixed(env_t *env, vsamples_t *vs, size_t D, gsl_matrix *c, int **m) {
  //printf("Running, D = %lu\n", D);
  size_t dimension = env->samples->dimension;
  int all_zero = 1;
  for(size_t class = 0; class < env->samples->class_cnt; class++) {
    for(size_t i = 0; i < env->samples->count[class]; i++) {
      gsl_vector *cs = apply_proj(c, vs->samples[class][i]);
      if(!is_zero_vec(cs)) {
	all_zero = 0;
	gsl_vector_free(cs);
	break;
      }
      gsl_vector_free(cs);
    }
    if(!all_zero) break;
  }
  if(all_zero) {
    printf("all zero\n");
    return gsl_vector_calloc(dimension);
  }
  if(D == 1) {
    gsl_vector *u1 = gsl_vector_calloc(dimension);
    gsl_vector_set_all(u1, 1);
    gsl_vector *v1 = apply_proj(c, u1);
    gsl_vector *v2 = gsl_vector_calloc(dimension);
    gsl_vector_axpby(-1, v1, 0, v2); 
    gsl_vector_free(u1);
    double obj1 = compute_obj_mixed(env, vs, v1, c, m).obj;
    double obj2 = compute_obj_mixed(env, vs, v2, c, m).obj;
    //printf("obj1 = %g, obj2 = %g\n", obj1, obj2);
    if(obj1 > obj2) {
      gsl_vector_free(v2);
      return v1;
    } else {
      gsl_vector_free(v1);
      return v2;
    }
  }

  double best_obj_case1 = -1e101;
  gsl_vector *best_w_case1 = NULL;
  double best_obj_case2 = -1e101;
  gsl_vector *best_w_case2 = NULL;
  gsl_vector *best_u_case2 = NULL;

  for(size_t class = 0; class < env->samples->class_cnt; class++) {
    for(size_t i=0; i < env->samples->count[class]; i++) {
      gsl_vector *u = apply_proj(c, vs->samples[class][i]);
      if(is_zero_vec(u)) {
	gsl_vector_free(u);
	continue;
      }
      //Not necessary to store V - we can get the basis vectors using c
      /*sparse_vector_t *newV = copy_sparse_vector(V);
	append(newV, class, i); //probably not a good idea to use sparse_vector for this; instead, use a regular vector and just store a pair*/
      gsl_matrix *newc = add_new_proj(c, u);
      gsl_vector *w_case1 = solve_exact_mixed(env, vs, D-1, newc, m);
      double obj_case1 = compute_obj_mixed(env, vs, w_case1, c, m).obj; //this could be eliminated by returning the objective from this function as well as the hyperplane
      if(obj_case1 > best_obj_case1) {
	best_obj_case1 = obj_case1;
	if(best_w_case1 != NULL)
	  gsl_vector_free(best_w_case1);
	best_w_case1 = w_case1;
      } else {
	gsl_vector_free(w_case1);
      }

      int **new_m = build_new_m(vs, c, u, m);
      gsl_vector *w_case2 = solve_exact_mixed(env, vs, D-1, newc, new_m);
      double obj_case2 = compute_obj_mixed(env, vs, w_case2, newc, new_m).obj;
      if(obj_case2 > best_obj_case2) {
	best_obj_case2 = obj_case2;
	if(best_w_case2 != NULL)
	  gsl_vector_free(best_w_case2);
	best_w_case2 = w_case2;
	if(best_u_case2 != NULL)
	  gsl_vector_free(best_u_case2);
	best_u_case2 = u;
      } else {
	gsl_vector_free(w_case2);
	gsl_vector_free(u);
      }
      gsl_matrix_free(newc);
    }
  }

  if(best_obj_case1 >= best_obj_case2) {
    gsl_vector_free(best_w_case2);
    gsl_vector_free(best_u_case2);
    printf("Case 1: %g\n", best_obj_case2);
    return best_w_case1;
  } else {
    printf("Case 2: %g\n", best_obj_case2);
    gsl_vector_free(best_w_case1);

    double alpha = 1e101;
    for(int class = 0; class < env->samples->class_cnt; class++) {
      for(int i = 0; i < env->samples->count[class]; i++) {
	double cand, denom;
	gsl_vector *cs = apply_proj(c, vs->samples[class][i]);
	gsl_blas_ddot(best_w_case2, cs, &cand);
	gsl_blas_ddot(best_u_case2, cs, &denom);
	cand *= -1/denom;
	if(cand < 0)
	  continue; //the only times that this would happen, any positive alpha works
	if(cand < alpha)
	  alpha = cand;
	gsl_vector_free(cs);
      }
    }
    alpha /= 2;
    printf("Chosen |alpha| = %g\n", alpha);
    gsl_vector *w_pos_shift = gsl_vector_alloc(dimension);
    gsl_vector *w_neg_shift = gsl_vector_alloc(dimension);
    gsl_vector_memcpy(w_pos_shift, best_w_case2);
    gsl_vector_axpby(alpha, best_u_case2, 1, w_pos_shift);
    gsl_vector_memcpy(w_neg_shift, best_w_case2);
    gsl_vector_axpby(-alpha, best_u_case2, 1, w_neg_shift);
    double obj_pos = compute_obj_mixed(env, vs, w_pos_shift, c, m).obj;
    double obj_neg = compute_obj_mixed(env, vs, w_neg_shift, c, m).obj;

    gsl_vector_free(best_u_case2);

    printf("pos: %g, neg: %g\n", obj_pos, obj_neg);
    
    if(obj_pos > obj_neg) {
      printf("chosen pos\n");
      gsl_vector_free(w_neg_shift);
      return w_pos_shift;
    } else {
      printf("chosen neg\n");
      gsl_vector_free(w_pos_shift);
      return w_neg_shift;
    }
  }

  /*printf("Stepping out. D = %lu, obj = %g\n", D, best_obj);
  printf(" -> best_w = ");
  for(size_t i = 0; i < dimension; i++)
    printf("%g%s", gsl_vector_get(best_proj_w, i), i == dimension - 1 ? "\n" : ", ");
    return best_w;*/
  return NULL;
}

void free_subproblem(subproblem_t *prob) {
  gsl_matrix_free(prob->c);
  free(prob);
}

cones_result_t solve_subproblem(env_t *env, vsamples_t *vs, prio_queue_t *probs, subproblem_t *prob, gsl_vector *best_w, pthread_mutex_t *prob_lock, round_pts_t *round_pts) {
  /** Solves a subproblem for the cones algorithm
   *  If it's a base case, returns the solution
   *  Otherwise, it adds its subproblems to the priority queue and returns NULL */
  if(prob_lock) pthread_mutex_lock(prob_lock);
  gsl_matrix *c = prob->c;
  size_t D = prob->D;
  if(prob_lock) pthread_mutex_unlock(prob_lock);
  //printf("Running, D = %lu\n", D);
  size_t dimension = vs->dimension;
  int all_zero = 1;

  for(size_t class = 0; class < vs->class_cnt; class++) {
    for(size_t i = 0; i < vs->count[class]; i++) {
      gsl_vector *s = vs->samples[class][i];
      gsl_vector *cs = apply_proj(c, s);
      //gsl_vector *cs = apply_proj(c, vs->samples[class][i]);
      if(!is_zero_vec(cs)) {
	all_zero = 0;
	gsl_vector_free(cs);
	break;
      }
      gsl_vector_free(cs);
    }
    if(!all_zero) break;
  }
  if(all_zero) {
    printf("all zero\n");
    return (cones_result_t) {.obj = positives(env->samples), .w = gsl_vector_calloc(dimension)};
  }
  if(D == 1) {
    gsl_vector *u1 = gsl_vector_calloc(dimension);
    gsl_vector_set_all(u1, 1);
    gsl_vector *v1 = apply_proj(c, u1);
    gsl_vector *v2 = gsl_vector_calloc(dimension);
    gsl_vector_axpby(-1, v1, 0, v2); 
    gsl_vector_free(u1);
    double obj1 = compute_obj(env, vs, v1, c).obj;
    double obj2 = compute_obj(env, vs, v2, c).obj;
    //printf("obj1 = %g, obj2 = %g\n", obj1, obj2);
    if(obj1 > obj2) {
      gsl_vector_free(v2);
      return (cones_result_t) {.obj = obj1, .w = v1};
    } else {
      gsl_vector_free(v1);
      return (cones_result_t) {.obj = obj2, .w = v2};
    }
  }

  double best_obj = -1e101;
  gsl_vector *best_proj_w = NULL;

  if(round_pts == NULL) {
    for(size_t class = prob->last_proj.class; class < env->samples->class_cnt; class++) {
      for(size_t i=prob->last_proj.idx+1; i < env->samples->count[class]; i++) {
	gsl_vector *u = apply_proj(c, vs->samples[class][i]);
	if(is_zero_vec(u)) {
	  gsl_vector_free(u);
	  continue;
	}
	//Not necessary to store V - we can get the basis vectors using c
	/*sparse_vector_t *newV = copy_sparse_vector(V);
	  append(newV, class, i); //probably not a good idea to use sparse_vector for this; instead, use a regular vector and just store a pair*/
	gsl_matrix *newc = add_new_proj(c, u);
	gsl_vector *proj_w = apply_proj(newc, best_w);
	subproblem_t *sub = CALLOC(1, subproblem_t);
	sub->D = D-1;
	sub->c = newc;
	sub->last_proj.class = class;
	sub->last_proj.idx = i;
	obj_result objres = compute_obj(env, vs, proj_w, newc);
	double obj = objres.obj;
	double prec = objres.prec;
	sub->score = obj;
	if(prob_lock) pthread_mutex_lock(prob_lock);
	prio_enqueue(probs, sub);
	if(prob_lock) pthread_mutex_unlock(prob_lock);
	gsl_vector_free(u);
      }
    }
  } else {
    for(int i = prob->last_proj.idx + 1; i < round_pts->n; i++) {
      gsl_vector *u = apply_proj(c, round_pts->pts[i]);
      if(is_zero_vec(u)) {
	gsl_vector_free(u);
	continue;
      }
      gsl_matrix *newc = add_new_proj(c, u);
      gsl_vector *proj_w = apply_proj(newc, best_w);
      subproblem_t *sub = CALLOC(1, subproblem_t);
      sub->D = D-1;
      sub->c = newc;
      sub->last_proj.class = -1;
      sub->last_proj.idx = i;
      obj_result objres = compute_obj(env, vs, proj_w, newc);
      double obj = objres.obj;
      double prec = objres.prec;
      sub->score = obj;
      //printf("about to enqueue\n");
      if(prob_lock) pthread_mutex_lock(prob_lock);
      prio_enqueue(probs, sub);
      if(prob_lock) pthread_mutex_unlock(prob_lock);
      //printf("enqueued\n");
      gsl_vector_free(u);
    }    
  }
  return (cones_result_t) {.obj = best_obj, .w = best_proj_w};    
}

gsl_vector *solve_exact_closed_pq(env_t *env, vsamples_t *vs, prio_queue_t *probs, int n_round_pts) {
  size_t d = env->samples->dimension;
  double *h0 = best_random_hyperplane_unbiased(1, env);
  //double *h0 = best_random_hyperplane_proj(1, env);
  gsl_vector *best_w = gsl_vector_calloc(d);
  *best_w = gsl_vector_view_array(h0, d).vector;
  double best_obj = compute_obj(env, vs, best_w, NULL).obj;
  printf("Random hyperplane has obj %g\n", best_obj);
  for(int i = 0; i <= d; i++) {
    printf("%g%s", h0[i], i < d ? ", " : "\n");
  }

  round_pts_t *round_pts = NULL;
  vsamples_t *rvs = NULL;
  if(n_round_pts > 0) {
    round_pts = CALLOC(1, round_pts_t);
    round_pts->n = n_round_pts;
    //round_pts->pts = uniform_sphere_points(n_round_pts, d);
    round_pts->pts = random_round_points(env, vs, n_round_pts);
    rvs = create_rounded_vs(vs, round_pts);
  }
  vsamples_t *old_vs = vs;
  if(rvs)
    vs = rvs;

  time_t start_time = time(NULL);
  long probs_solved = 0;
  int time_int = 0;
  
  while(prio_get_n(probs) > 0) {
    time_t curr_time = time(NULL);
    double dtime = difftime(curr_time, start_time);
    if(dtime > 120)
      break;
    if((int) dtime > time_int) {
      time_int = (int) dtime;
      printf("[%ds] %ld subproblems solved\n", time_int, probs_solved);
    }
    subproblem_t *prob = prio_pop_max(probs);
    cones_result_t res = solve_subproblem(env, vs, probs, prob, best_w, NULL, round_pts);
    probs_solved++;
    if(res.obj > best_obj) {
      printf("New best obj %g\n", res.obj);
      if(best_w)
	gsl_vector_free(best_w);
      best_obj = res.obj;
      best_w = res.w;
    } else {
      gsl_vector_free(res.w);
    }
    free_subproblem(prob);
  }
  //free(h0);
  return best_w;
}

typedef struct subprob_thread_t {
  env_t *env;
  vsamples_t *vs;
  prio_queue_t *probs;
  gsl_vector *best_w;
  double *best_obj;
  int *n_probs;
  round_pts_t *round_pts;
  int *stop;
  int *waiting;
  pthread_mutex_t *prob_lock;
  pthread_cond_t *prob_cond;
  pthread_mutex_t *obj_lock;
  pthread_mutex_t *wait_lock;
  pthread_cond_t *wait_cond;
  pthread_t *thread;
  int id;
} subprob_thread_t;

void *subproblem_worker(void *args) {
  subprob_thread_t *a = args;
  while(!(*a->stop)) { //reading an int should be atomic, so this shouldn't need a mutex
    pthread_mutex_lock(a->wait_lock);
    *a->waiting = 1;
    pthread_mutex_unlock(a->wait_lock);
    
    pthread_mutex_lock(a->prob_lock);
    while(prio_get_n(a->probs) == 0 && !(*a->stop)) {
      pthread_cond_wait(a->prob_cond, a->prob_lock);
    }
    pthread_mutex_lock(a->wait_lock);
    *a->waiting = 0;
    pthread_cond_signal(a->wait_cond);
    pthread_mutex_unlock(a->wait_lock);
    
    if(*a->stop)
      break;
    subproblem_t *prob = prio_pop_max(a->probs);
    pthread_mutex_unlock(a->prob_lock);
    
    cones_result_t res = solve_subproblem(a->env, a->vs, a->probs, prob, a->best_w,a->prob_lock, a->round_pts);
    
    pthread_mutex_lock(a->prob_lock);
    pthread_cond_signal(a->prob_cond);
    pthread_mutex_unlock(a->prob_lock);

    pthread_mutex_lock(a->obj_lock);
    *a->n_probs += 1;
    if(res.obj > *a->best_obj) {
      printf("[T%d] New best solution with objective %g\n", a->id, res.obj);
      gsl_vector_memcpy(a->best_w, res.w);
      *a->best_obj = res.obj;
    }
    pthread_mutex_unlock(a->obj_lock);


    free(prob);
  }
  return NULL;
}

int all_waiting(subprob_thread_t *workers) {
  //checks if all workers are waiting
  //this function assumes you've acquired the wait lock
  for(int i = 0; i < NUM_THREADS; i++) {
    if(!(*workers[i].waiting)) return 0;
  }
  return 1;
}

gsl_vector *solve_exact_closed_pq_mt2(env_t *env, vsamples_t *vs, prio_queue_t *probs, int n_round_pts) {
  size_t d = env->samples->dimension;
  double *h0 = best_random_hyperplane_unbiased(1, env);
  //double *h0 = CALLOC(d, double);
  gsl_vector *best_w = gsl_vector_calloc(d);
  *best_w = gsl_vector_view_array(h0, d).vector;
  double best_obj = compute_obj(env, vs, best_w, NULL).obj;
  printf("Random hyperplane has obj %g\n", best_obj);
  for(int i = 0; i <= d; i++) {
    printf("%g%s", h0[i], i < d ? ", " : "\n");
  }
  
  round_pts_t *round_pts = NULL;
  vsamples_t *rvs = NULL;
  if(n_round_pts > 0) {
    round_pts = CALLOC(1, round_pts_t);
    round_pts->n = n_round_pts;
    //round_pts->pts = uniform_sphere_points(n_round_pts, d);
    round_pts->pts = random_round_points(env, vs, n_round_pts);
    rvs = create_rounded_vs(vs, round_pts);
  }
  vsamples_t *old_vs = vs;
  if(rvs)
    vs = rvs;
  
  subprob_thread_t workers[NUM_THREADS];
  pthread_mutex_t prob_lock;
  pthread_mutex_init(&prob_lock, NULL);
  pthread_cond_t prob_cond;
  pthread_cond_init(&prob_cond, NULL);
  pthread_mutex_t obj_lock;
  pthread_mutex_init(&obj_lock, NULL);
  pthread_mutex_t wait_lock;
  pthread_mutex_init(&wait_lock, NULL);
  pthread_cond_t wait_cond;
  pthread_cond_init(&wait_cond, NULL);

  double *best_obj_univ = CALLOC(1, double);
  *best_obj_univ = best_obj;
  int *n_probs = CALLOC(1, int);
  

  for(int i = 0; i < NUM_THREADS; i++) {
    workers[i].env = env;
    workers[i].vs = vs;
    workers[i].probs = probs;
    workers[i].best_w = best_w;
    workers[i].best_obj = best_obj_univ;
    workers[i].n_probs = n_probs;
    workers[i].round_pts = round_pts;
    workers[i].stop = CALLOC(1, int);
    workers[i].waiting = CALLOC(1, int);
    workers[i].prob_lock = &prob_lock;
    workers[i].prob_cond = &prob_cond;
    workers[i].obj_lock = &obj_lock;
    workers[i].wait_lock = &wait_lock;
    workers[i].wait_cond = &wait_cond;
    workers[i].thread = CALLOC(1, pthread_t);
    workers[i].id = i;
    pthread_create(workers[i].thread, NULL, subproblem_worker, &workers[i]);
  }

  time_t start_time = time(NULL);
  int time_int = 0;
  
  //wait until all threads are done and prio queue is empty
  pthread_mutex_lock(&wait_lock);
  //printf("main thread got wait lock\n");
  while(prio_get_n(probs) > 0 || !all_waiting(workers)) {
    //printf("main thread waiting\n");
    pthread_cond_wait(&wait_cond, &wait_lock);
    time_t curr_time = time(NULL);
    double dtime = difftime(curr_time, start_time);
    if((int) dtime > time_int) {
      time_int = (int) dtime;
      pthread_mutex_lock(&obj_lock);
      printf("[%ds] %d subproblems solved\n", time_int, *n_probs);
      pthread_mutex_unlock(&obj_lock);
    }
  }
  //now, all threads are done, so tell them to stop
  printf("# unsolved probs = %ld. All waiting? %d\n", prio_get_n(probs), all_waiting(workers));
  for(int i = 0; i < NUM_THREADS; i++)
    *workers[i].stop = 1;
  pthread_mutex_unlock(&wait_lock);
  pthread_mutex_lock(&prob_lock);
  pthread_cond_signal(&prob_cond);
  pthread_mutex_unlock(&prob_lock);

  printf("best objective found: %g\n", best_obj);
  printf("joining threads...\n");
  for(int i = 0; i < NUM_THREADS; i++)
    pthread_join(*workers[i].thread, NULL);
  printf("joined\n");

  //free(h0);
  return best_w;
}

typedef struct subprob_args_t {
  env_t *env;
  vsamples_t *vs;
  prio_queue_t *probs;
  subproblem_t *prob;
  gsl_vector *best_w;
  double *best_obj;
  cones_result_t *res;
  round_pts_t *round_pts;
  int *completed;
  int *nrunning;
  pthread_mutex_t *completed_lock;
  pthread_cond_t *completed_cond;
  int *stop;
  pthread_mutex_t *prob_lock;
  pthread_cond_t *prob_cond;
  pthread_mutex_t *obj_lock;
} subprob_args_t;

void *solve_subproblem_mt(void *args) {
  subprob_args_t *a = args;
  printf("about to solve subproblem\n");
  cones_result_t res = solve_subproblem(a->env, a->vs, a->probs, a->prob, a->best_w, NULL, a->round_pts);
  printf("solved\n");
  memcpy(a->res, &res, sizeof(res));
  pthread_mutex_t *lock = a->completed_lock;
  printf("about to acquire lock\n");
  pthread_mutex_lock(lock);
  *a->completed = 1;
  *a->nrunning -= 1;
  printf("set completed and nrunning\n");
  pthread_cond_signal(a->completed_cond);
  printf("thread done\n");
  free(args);
  free(a->prob);
  pthread_mutex_unlock(lock);
  return NULL;
}

gsl_vector *solve_exact_closed_pq_mt(env_t *env, vsamples_t *vs, prio_queue_t *probs, int n_round_pts) {
  size_t d = env->samples->dimension;
  double *h0 = best_random_hyperplane_unbiased(1, env);
  //double *h0 = best_random_hyperplane_proj(1, env);
  gsl_vector *best_w = gsl_vector_calloc(d);
  *best_w = gsl_vector_view_array(h0, d).vector;
  double best_obj = compute_obj(env, vs, best_w, NULL).obj;
  printf("Random hyperplane has obj %g\n", best_obj);
  for(int i = 0; i <= d; i++) {
    printf("%g%s", h0[i], i < d ? ", " : "\n");
  }
  
  round_pts_t *round_pts = NULL;
  vsamples_t *rvs = NULL;
  if(n_round_pts > 0) {
    round_pts = CALLOC(1, round_pts_t);
    round_pts->n = n_round_pts;
    //round_pts->pts = uniform_sphere_points(n_round_pts, d);
    round_pts->pts = random_round_points(env, vs, n_round_pts);
    rvs = create_rounded_vs(vs, round_pts);
  }
  vsamples_t *old_vs = vs;
  if(rvs)
    vs = rvs;
  
  pthread_t threads[NUM_THREADS];
  cones_result_t *results = CALLOC(NUM_THREADS, cones_result_t);
  int *completed_threads = CALLOC(NUM_THREADS, int);
  int *nrunning = CALLOC(1, int);
  int started = 0;
  for(int i = 0; i < NUM_THREADS; i++) completed_threads[i] = -1;
  int next_thread = 0;

  pthread_mutex_t completed_lock;
  pthread_cond_t completed_cond;
  pthread_mutex_init(&completed_lock, NULL);
  pthread_cond_init(&completed_cond, NULL);
  while(prio_get_n(probs) > 0 || (*nrunning > 0 || !started)) {
    printf("# probs = %lu, running = %d, next_thread = %d\n", prio_get_n(probs), *nrunning, next_thread);
    printf("thread status: ");
    for(int i = 0; i < NUM_THREADS; i++)
      printf("%d%s", completed_threads[i], i == NUM_THREADS-1 ? "\n": " ");
    //check that nrunning < NUM_THREADS
    printf("about to acquire lock to seek\n");
    pthread_mutex_lock(&completed_lock);
    printf("acquired lock to seek\n");
    while(*nrunning >= NUM_THREADS) {
      printf("waiting for signal\n");
      pthread_cond_wait(&completed_cond, &completed_lock);
    }
    printf("now nrunning = %d\n", *nrunning);
    
    while(completed_threads[next_thread] == 0) {
      //seek for next available thread
      next_thread++;
      if(next_thread == NUM_THREADS)
	next_thread = 0;
    }
    
    if(completed_threads[next_thread] == 1) {
      //this indicates that a thread completed, so join it first
      printf("about to join\n");
      pthread_join(threads[next_thread], NULL); //can this cause a deadlock, if this thread is waiting on the lock? No, because it should be done already
      //*nrunning -= 1;
      completed_threads[next_thread] = -1;
      pthread_mutex_unlock(&completed_lock);
      cones_result_t res = results[next_thread];
      if(res.obj > best_obj) {
	printf("New best obj %g\n", res.obj);
	if(best_w)
	  gsl_vector_free(best_w);
	best_obj = res.obj;
	best_w = res.w;
      } else {
	gsl_vector_free(res.w);
      }
    } else {
      pthread_mutex_unlock(&completed_lock);
    }
    subproblem_t *prob = prio_pop_max(probs);
    //printf("popped problem\n");
    if(prob == NULL) {
      //printf("problem is null\n");
      next_thread++;
      if(next_thread == NUM_THREADS)
	next_thread = 0;
      continue;
    }
    printf("about to construct args\n");
    printf("nrunning = %d, thread status: ", *nrunning);
    for(int i = 0; i < NUM_THREADS; i++)
      printf("%d%s", completed_threads[i], i == NUM_THREADS-1 ? "\n": " ");
    
    subprob_args_t *args = CALLOC(1, subprob_args_t);
    args->env = env;
    args->vs = vs;
    args->probs = probs;
    args->prob = prob;
    args->best_w = best_w;
    args->res = &results[next_thread];
    args->completed = &completed_threads[next_thread];
    args->round_pts = round_pts;
    args->nrunning = nrunning;
    args->completed_lock = &completed_lock;
    args->completed_cond = &completed_cond;
    //printf("about to create thread\n");
    pthread_create(&threads[next_thread], NULL, solve_subproblem_mt, args);
    //printf("created thread\n");
    started = 1;
    pthread_mutex_lock(&completed_lock);
    completed_threads[next_thread] = 0;
    *nrunning += 1;
    pthread_mutex_unlock(&completed_lock);
  }
  printf("nrunning = %d\n", *nrunning);
  //free(h0);
  return best_w;
}

double *single_exact_run(env_t *env, int n_round_pts) {
  size_t dimension = env->samples->dimension;
  //sparse_vector_t *V = sparse_vector_blank(dimension+1);
  gsl_matrix *c = gsl_matrix_calloc(dimension, dimension);
  for(size_t i = 0; i < dimension; i++)
    gsl_matrix_set(c, i, i, 1);
  //double *w = solve_exact_closed(env, env->samples->dimension, V, c);
  vsamples_t *vs = samples_to_vec(env->samples);
  //gsl_vector *w = solve_exact_closed(env, vs, env->samples->dimension, c);

  prio_queue_t *probs = create_prio_queue(1);
  subproblem_t *base = CALLOC(1, subproblem_t);
  base->D = dimension;
  base->c = c;
  base->score = 1;
  base->last_proj.class = 0;
  base->last_proj.idx = -1;
  prio_enqueue(probs, base);
  gsl_vector *w = solve_exact_closed_pq_mt2(env, vs, probs, n_round_pts);
  //gsl_vector *w = solve_exact_closed(env, vs, dimension, c);
  printf("obj = %g\n", compute_obj(env, vs, w, NULL).obj);
  double *h = CALLOC(dimension, double);
  for(size_t i = 0; i < dimension; i++)
    h[i] = gsl_vector_get(w, i);
  gsl_vector_free(w);
  vsamples_free(vs);
  return h;
}

double *single_exact_run_open(env_t *env) {
  size_t dimension = env->samples->dimension;
  gsl_matrix *c = gsl_matrix_calloc(dimension, dimension);
  for(size_t i = 0; i < dimension; i++)
    gsl_matrix_set(c, i, i, 1);

  vsamples_t *vs = samples_to_vec(env->samples);
  int **m = CALLOC(env->samples->class_cnt, int *);
  for(int class = 0; class < env->samples->class_cnt; class++) {
    m[class] = CALLOC(env->samples->count[class], int);
    for(int i = 0; i < env->samples->count[class]; i++) {
      m[class][i] = 1;
    }
  }

  
  gsl_vector *w = solve_exact_mixed(env, vs, env->samples->dimension, c, m);
  printf("obj = %g\n", compute_obj(env, vs, w, NULL).obj);
  double *h = CALLOC(dimension, double);
  for(size_t i = 0; i < dimension; i++)
    h[i] = gsl_vector_get(w, i);
  gsl_vector_free(w);
  vsamples_free(vs);
  return h;
}

struct rand_proj_res {
  gsl_vector *w;
  gsl_matrix *c;
  int *indices;
} best_random_proj(int initial, env_t *env) {
  params_t *params = env->params;
  int rnd_trials = initial ? params->rnd_trials : params->rnd_trials_cont;
  if (!rnd_trials) {
    return (struct rand_proj_res) {0};
  }
  
  samples_t *samples = env->samples;
  vsamples_t *vs = samples_to_vec(samples);
  size_t d = samples->dimension;
  double best_value = -1e101;
  int *best_pts = NULL;
  gsl_matrix *best_c = NULL;
  gsl_vector *best_hyperplane = NULL;
  int n = samples_total(samples);
  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(r, rand());
  
  for(int k = 0; k < rnd_trials; k++) {
    int *pts = CALLOC(d-1, int);
    for(int i = 0; i < d-1; i++) {
      //generate the d - 1 random indices
      //TODO: find a better way to do this (maybe a library function
      int repeat;
      do {
	repeat = 0;
	int newpt = gsl_rng_uniform_int(r, env->samples->dimension);
	for(int j = 0; j < i; j++) {
	  if(pts[i] == newpt) {
	    repeat = 1;
	    break;
	  }
	}
      } while(repeat);
    }

    gsl_matrix *c = gsl_matrix_alloc(d, d);
    for(size_t i = 0; i < d; i++)
      gsl_matrix_set(c, i, i, 1);
    for(int i = 0; i < d-1; i++) {
      sample_locator_t *loc = locator(pts[i]+d+2, env->samples);
      gsl_vector *u = vs->samples[loc->class][loc->index];
      gsl_vector *cu = apply_proj(c, u);
      add_new_proj_ip(c, cu);
      gsl_vector_free(cu);
      free(loc);
    }
    gsl_vector *u1 = gsl_vector_calloc(d);
    gsl_vector_set_all(u1, 1);
    gsl_vector *v1 = apply_proj(c, u1);
    gsl_vector *v2 = gsl_vector_calloc(d);
    gsl_vector_axpby(-1, v1, 0, v2); 
    gsl_vector_free(u1);
    double obj1 = compute_obj(env, vs, v1, c).obj;
    double obj2 = compute_obj(env, vs, v2, c).obj;
    //printf("obj1 = %g, obj2 = %g\n", obj1, obj2);

    gsl_vector *w;
    int obj;
    if(obj1 > obj2) {
      gsl_vector_free(v2);
      w = v1;
      obj = obj1;
    } else {
      gsl_vector_free(v1);
      w = v2;
      obj = obj2;
    }
    if(obj > best_value) {
      if(best_pts) {
	free(best_pts);
	gsl_vector_free(best_hyperplane);
	gsl_matrix_free(best_c);
      }
      best_hyperplane = w;
      best_value = obj;
      best_c = c;
      best_pts = pts;
    } else {
      gsl_vector_free(w);
      gsl_matrix_free(c);
      free(pts);
    }
  }
  vsamples_free(vs);
  gsl_rng_free(r);
  printf("best obj = %g\n", best_value);
  return (struct rand_proj_res) {.w = best_hyperplane, .c = best_c, .indices = best_pts};
}

double *best_random_hyperplane_proj(int initial, env_t *env) {
  struct rand_proj_res res = best_random_proj(initial, env);
  gsl_matrix_free(res.c);
  free(res.indices);
  double *h = CALLOC(env->samples->dimension, double);
  for(int i = 0; i < env->samples->dimension; i++)
    h[i] = gsl_vector_get(res.w, i);
  return h;
}

gsl_vector *get_proj_basis(env_t *env, gsl_vector **V) {
  //find the basis vector given by sequentially projecting onto each hyperplane in V. assumes length(V) = d-1
  //I think this assumes that the vectors in V are linearly independent
  size_t d = env->samples->dimension;
  gsl_matrix *A = gsl_matrix_calloc(d, d-1);
  for(int i = 0; i < d-1; i++) {
    gsl_matrix_set_col(A, i, V[i]);
  }
  gsl_matrix *T = gsl_matrix_alloc(d-1, d-1);
  gsl_linalg_QR_decomp_r(A, T);
  //final column of Q is the sole null-space vector of A, assuming A had full rank
  gsl_matrix *Q = gsl_matrix_alloc(d, d);
  gsl_matrix *R = gsl_matrix_alloc(d-1, d-1);
  gsl_linalg_QR_unpack_r(A, T, Q, R);

  gsl_vector *v = gsl_vector_alloc(d);
  gsl_matrix_get_col(v, Q, d-1);

  gsl_matrix_free(A);
  gsl_matrix_free(T);
  gsl_matrix_free(Q);
  gsl_matrix_free(R);
  return v;
}
