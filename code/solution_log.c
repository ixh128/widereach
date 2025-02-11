#include "general.h"
#include "helper.h"

solution_log_t *solution_log_init(size_t init_cap) {
  solution_log_t *l = CALLOC(1, solution_log_t);
  l->cap = init_cap;
  l->n_pts = 0;
  l->objs = CALLOC(l->cap, double);
  l->times = CALLOC(l->cap, double);

  return l;
}

solution_log_t *delete_solution_log(solution_log_t *log) {
  free(log->objs);
  free(log->times);
  return log;
}

void append_to_log(solution_log_t *log, double obj, double time) {
  double **objs = &log->objs;
  double **times = &log->times;
  size_t *n_pts = &log->n_pts;
  size_t *cap = &log->cap;

  if (*objs == NULL || *times == NULL) {
    printf("ERR: solution log not correctly initialized\n");
  }

  if (*n_pts == *cap) {
    *cap *= 2;
    *objs = realloc(*objs, *cap * sizeof(double));
    *times = realloc(*times, *cap * sizeof(double));
    if (*objs == NULL || *times == NULL) {
      printf("a realloc failed\n");
    }
  }

  (*objs)[*n_pts] = obj;
  (*times)[*n_pts] = time;
  *n_pts += 1;
}

void print_solution_log(solution_log_t *log) {
  printf("   t      |   obj   \n");
  printf("--------------------\n");

  for (int i = 0; i < log->n_pts; i++) {
    printf("%8.3f, %8.3f,\n", log->times[i], log->objs[i]);
  }
}
