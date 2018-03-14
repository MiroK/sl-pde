#line 1 "probing-cpp.c"
#line 1 "<built-in>"
#line 1 "<command-line>"
#line 1 "/usr/include/stdc-predef.h"
#line 1 "<command-line>"
#line 1 "probing-cpp.c"
#if _XOPEN_SOURCE < 700
#undef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif
#if _GNU_SOURCE
#include <stdint.h>
#include <string.h>
#include <fenv.h>
#endif


#line 1 "/home/miro3/Documents/Programming/basilisk/src/common.h"
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdarg.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

#if _CADNA
# include <cadna.h>
#endif

#if __cplusplus
# define delete delete_qcc
# define right right_qcc
# define left left_qcc
# define norm norm_qcc
# define new new_qcc
#endif

#define pi 3.14159265358979
#undef HUGE
#define HUGE ((double)1e30)
#define nodata HUGE
#define _NVARMAX 65536
#define is_constant(v) ((v).i >= _NVARMAX)
#define constant(v) (is_constant(v) ? _constant[(v).i - _NVARMAX] : nodata)

#define max(a,b) ((a) > (b) ? (a) : (b))
#define min(a,b) ((a) < (b) ? (a) : (b))
#define sq(x) ((x)*(x))
#define cube(x) ((x)*(x)*(x))
#define sign(x) ((x) > 0 ? 1 : -1)
#define noise() (1. - 2.*rand()/(double)RAND_MAX)
#define clamp(x,a,b) ((x) < (a) ? (a) : (x) > (b) ? (b) : (x))
#define swap(type,a,b) { type tmp = a; a = b; b = tmp; }
#define unmap(x,y)

#define trash(x)


#define systderr stderr
#define systdout stdout

#if 1
static FILE * qstderr (void);
static FILE * qstdout (void);
FILE * ferr, * fout;
#define not_mpi_compatible()\
do {\
  if (npe() > 1) {\
    fprintf (ferr, "%s() is not compatible with MPI (yet)\n", __func__);\
    exit (1);\
  }\
} while(0)\

#line 62

#else
# define qstderr() stderr
# define qstdout() stdout
# define ferr stderr
# define fout stdout
# define not_mpi_compatible()
#endif



#if MTRACE

struct {
  FILE * fp;
  size_t total, max;
  size_t overhead, maxoverhead;
  size_t nr;
  size_t startrss, maxrss;
  char * fname;
} pmtrace;

typedef struct {
  char * func, * file;
  size_t max, total;
  int line, id;
} pmfunc;

typedef struct {
  size_t id, size;
} pmdata;

static pmfunc * pmfuncs = NULL;
static int pmfuncn = 0;

#define sysmalloc malloc
#define syscalloc calloc
#define sysrealloc realloc
#define sysfree free
#define systrdup strdup

static int pmfunc_index (const char * func, const char * file, int line)
{
  pmfunc * p = pmfuncs;
  for (int i = 0; i < pmfuncn; i++, p++)
    if (p->line == line && !strcmp(func, p->func) && !strcmp(file, p->file))
      return p->id;
  pmfuncn++;
  pmfuncs = (pmfunc *) sysrealloc (pmfuncs, pmfuncn*sizeof(pmfunc));
  p = &pmfuncs[pmfuncn - 1];
  memset (p, 0, sizeof(pmfunc));
  p->func = systrdup(func);
  p->file = systrdup(file);
  p->line = line;
  p->id = pmfuncn;
  if (pmtrace.fp)
    fprintf (pmtrace.fp, "@ %d %s %s %d\n", pmfuncn, func, file, line);
  return pmfuncn;
}

static void pmfunc_trace (pmfunc * f, char c)
{
  if (pmtrace.fp)
    fprintf (pmtrace.fp, "%c %d %ld %ld %ld",
      c, f->id, pmtrace.nr, pmtrace.total, f->total);
#if _GNU_SOURCE
  if (pmtrace.nr % 1 == 0) {
    struct rusage usage;
    getrusage (RUSAGE_SELF, &usage);
    if (pmtrace.fp)
      fprintf (pmtrace.fp, " %ld", usage.ru_maxrss*1024);
    if (!pmtrace.nr)
      pmtrace.startrss = usage.ru_maxrss;
    if (usage.ru_maxrss > pmtrace.maxrss)
      pmtrace.maxrss = usage.ru_maxrss;
  }
#endif
  if (pmtrace.fp)
    fputc ('\n', pmtrace.fp);
  pmtrace.nr++;
}

static void * pmfunc_alloc (pmdata * d, size_t size,
       const char * func, const char * file, int line,
       char c)
{
  assert (d != NULL);
  d->id = pmfunc_index(func, file, line);
  d->size = size;
  pmfunc * f = &pmfuncs[d->id - 1];
  f->total += size;
  if (f->total > f->max)
    f->max = f->total;
  pmtrace.total += size;
  pmtrace.overhead += sizeof(pmdata);
  if (pmtrace.total > pmtrace.max) {
    pmtrace.max = pmtrace.total;
    pmtrace.maxoverhead = pmtrace.overhead;
  }
  pmfunc_trace (f, c);
  return ((char *)d) + sizeof(pmdata);
}

static void * pmfunc_free (void * ptr, char c)
{
  if (!ptr)
    return ptr;
  pmdata * d = (pmdata *) (((char *)ptr) - sizeof(pmdata));
  if (d->id < 1 || d->id > pmfuncn) {
    fputs ("*** MTRACE: ERROR!: corrupted free()", qstderr());
    if (d->size == 0)
      fputs (", possible double free()", qstderr());
    else
      fputs (", not traced?", qstderr());
    fputs (", aborting...\n", qstderr());
    abort();
    return ptr;
  }
  else {
    pmfunc * f = &pmfuncs[d->id - 1];
    if (f->total < d->size) {
      fprintf (qstderr(), "*** MTRACE: ERROR!: %ld < %ld: corrupted free()?\n",
        f->total, d->size);
      abort();
    }
    else
      f->total -= d->size;
    if (pmtrace.total < d->size) {
      fprintf (qstderr(), "*** MTRACE: ERROR!: %ld < %ld: corrupted free()?\n",
        pmtrace.total, d->size);
      abort();
    }
    else {
      pmtrace.total -= d->size;
      pmtrace.overhead -= sizeof(pmdata);
    }
    d->id = 0;
    d->size = 0;
    pmfunc_trace (f, c);
    return d;
  }
}

static void * pmalloc (size_t size,
         const char * func, const char * file, int line)
{
  return pmfunc_alloc ((pmdata *) sysmalloc (sizeof(pmdata) + size),
         size, func, file, line, '+');
}

static void * pcalloc (size_t nmemb, size_t size,
         const char * func, const char * file, int line)
{
  void * p = pmalloc (nmemb*size, func, file, line);
  return memset (p, 0, nmemb*size);
}

static void * prealloc (void * ptr, size_t size,
   const char * func, const char * file, int line)
{
  return pmfunc_alloc ((pmdata *) sysrealloc (pmfunc_free(ptr, '<'),
           sizeof(pmdata) + size),
         size, func, file, line, '>');
}

static void pfree (void * ptr,
     const char * func, const char * file, int line)
{
  sysfree (pmfunc_free (ptr, '-'));
}

static char * pstrdup (const char * s,
         const char * func, const char * file, int line)
{
  char * d = (char *) pmalloc (strlen(s) + 1, func, file, line);
  return strcpy (d, s);
}

#if MTRACE < 3
static int pmaxsort (const void * a, const void * b) {
  const pmfunc * p1 = a, * p2 = b;
  return p1->max < p2->max;
}
#endif

static int ptotalsort (const void * a, const void * b) {
  const pmfunc * p1 = (const pmfunc *) a, * p2 = (const pmfunc *) b;
  return p1->total < p2->total;
}

static void pmfuncs_free()
{
  pmfunc * p = pmfuncs;
  for (int i = 0; i < pmfuncn; i++, p++) {
    sysfree (p->func);
    sysfree (p->file);
  }
  sysfree (pmfuncs);
}

void pmuntrace (void)
{
#if MTRACE < 3
  fprintf (qstderr(),
    "*** MTRACE: max resident  set size: %10ld bytes\n"
    "*** MTRACE: max traced memory size: %10ld bytes"
    " (tracing overhead %.1g%%)\n"
    "%10s    %20s   %s\n",
    pmtrace.maxrss*1024,
    pmtrace.max, pmtrace.maxoverhead*100./pmtrace.max,
    "max bytes", "function", "file");
  qsort (pmfuncs, pmfuncn, sizeof(pmfunc), pmaxsort);
  pmfunc * p = pmfuncs;
  for (int i = 0; i < pmfuncn && p->max > 0; i++, p++)
    fprintf (qstderr(), "%10ld    %20s   %s:%d\n",
      p->max, p->func, p->file, p->line);

  if (pmtrace.fp) {
    char * fname = pmtrace.fname, * s;
    while ((s = strchr(fname,'/')))
      fname = s + 1;

    fputs ("load(\"`echo $BASILISK`/mtrace.plot\")\n", pmtrace.fp);
    fprintf (pmtrace.fp,
      "plot '%s' u 3:($6-%g) w l t 'ru_maxrss - %.3g',"
      "total(\"%s\") w l t 'total'",
      fname,
      pmtrace.startrss*1024.,
      pmtrace.startrss*1024.,
      fname);
    pmfunc * p = pmfuncs;
    for (int i = 0; i < pmfuncn && p->max > 0.01*pmtrace.max; i++, p++)
      fprintf (pmtrace.fp,
        ",func(\"%s\",%d) w l t '%s'",
        fname, p->id, p->func);
    fputc ('\n', pmtrace.fp);
    fprintf (qstderr(),
      "*** MTRACE: To get a graph use: tail -n 2 %s | gnuplot -persist\n",
      fname);
    fclose (pmtrace.fp);
    pmtrace.fp = NULL;
    sysfree (pmtrace.fname);
  }
#endif

  if (pmtrace.total > 0) {
    qsort (pmfuncs, pmfuncn, sizeof(pmfunc), ptotalsort);
    pmfunc * p = pmfuncs;
    for (int i = 0; i < pmfuncn && p->total > 0; i++, p++)
      fprintf (qstderr(), "%s:%d: error: %ld bytes leaked here\n",
        p->file, p->line, p->total);
    pmfuncs_free();
    exit(1);
  }
  else {
#if MTRACE < 3
    fputs ("*** MTRACE: No memory leaks\n", qstderr());
#endif
    pmfuncs_free();
  }
}

#else
# define pmalloc(s,func,file,line) malloc(s)
# define pcalloc(n,s,func,file,line) calloc(n,s)
# define prealloc(p,s,func,file,line) realloc(p,s)
# define pfree(p,func,file,line) free(p)
# define pstrdup(s,func,file,line) strdup(s)
#endif







typedef struct {
  void * p;
  long max, len;
} Array;

Array * array_new()
{
  Array * a = ((Array *) pmalloc ((1)*sizeof(Array),__func__,__FILE__,__LINE__));
  a->p = NULL;
  a->max = a->len = 0;
  return a;
}

void array_free (Array * a)
{
  if (a->max > 0)
    pfree (a->p,__func__,__FILE__,__LINE__);
  pfree (a,__func__,__FILE__,__LINE__);
}

void array_append (Array * a, void * elem, size_t size)
{
  if (a->len + size >= a->max) {
    a->max += max (size, 4096);
    a->p = prealloc (a->p, a->max,__func__,__FILE__,__LINE__);
  }
  memcpy (((char *)a->p) + a->len, elem, size);
  a->len += size;
}

void * array_shrink (Array * a)
{
  void * p = prealloc (a->p, a->len,__func__,__FILE__,__LINE__);
  pfree (a,__func__,__FILE__,__LINE__);
  return p;
}



#if 2 == 1
#include <extrae_user_events.h>

typedef struct {
  Array index, stack;
  extrae_type_t type;
} Trace;

Trace trace_func = {
  {NULL, 0, 0}, {NULL, 0, 0},
  60000010,
};

Trace trace_mpi_func = {
  {NULL, 0, 0}, {NULL, 0, 0},
  60000011,
};

static int lookup_func (Array * a, const char * func)
{
  for (int i = 0; i < a->len/sizeof(char *); i++) {
    char * s = ((char **)a->p)[i];
    if (!strcmp (func, s))
      return i + 1;
  }
  char * s = pstrdup (func,__func__,__FILE__,__LINE__);
  array_append (a, &s, sizeof(char *));
  return a->len;
}

static void trace_push (Trace * t, const char * func)
{
  int value = lookup_func (&t->index, func);
  Extrae_eventandcounters (t->type, value);
  array_append (&t->stack, &value, sizeof(int));
}

static void trace_pop (Trace * t, const char * func)
{
  assert (t->stack.len > 0);
  t->stack.len -= sizeof(int);
  int value = t->stack.len > 0 ?
    ((int *)t->stack.p)[t->stack.len/sizeof(int) - 1] : 0;
  Extrae_eventandcounters (t->type, value);
}

static void trace_define (Trace * t, char * description)
{
  if (t->index.len > 0) {
    extrae_value_t values[t->index.len/sizeof(char *) + 1];
    char * names[t->index.len/sizeof(char *) + 1],
      ** func = (char **) t->index.p;
    names[0] = "OTHER";
    values[0] = 0;
    unsigned len = 1;
    for (int i = 0; i < t->index.len/sizeof(char *); i++, func++) {
      names[len] = *func;
      values[len++] = i + 1;
    }
    Extrae_define_event_type (&t->type, description, &len, values, names);
  }
}

static void trace_free (Trace * t)
{
  char ** func = (char **) t->index.p;
  for (int i = 0; i < t->index.len/sizeof(char *); i++, func++)
    pfree (*func,__func__,__FILE__,__LINE__);
  pfree (t->index.p,__func__,__FILE__,__LINE__);
  pfree (t->stack.p,__func__,__FILE__,__LINE__);
}

static void trace_off()
{
  trace_define (&trace_func, "Basilisk functions");
  trace_define (&trace_mpi_func, "Basilisk functions (MPI-related)");
  trace_free (&trace_func);
  trace_free (&trace_mpi_func);
}






# define trace(func, file, line) trace_push (&trace_func, func)
# define end_trace(func, file, line) trace_pop (&trace_func, func)

#elif 2

typedef struct {
  char * func, * file;
  int line, calls;
  double total, self;
} TraceIndex;

struct {
  Array stack, index;
  double t0;
} Trace = {
  {NULL, 0, 0}, {NULL, 0, 0},
  -1
};

static void trace_add (const char * func, const char * file, int line,
         double total, double self)
{
  TraceIndex * t = (TraceIndex *) Trace.index.p;
  int i, len = Trace.index.len/sizeof(TraceIndex);
  for (i = 0; i < len; i++, t++)
    if (t->line == line && !strcmp (func, t->func) && !strcmp (file, t->file))
      break;
  if (i == len) {
    TraceIndex t = {pstrdup(func,__func__,__FILE__,__LINE__), pstrdup(file,__func__,__FILE__,__LINE__), line, 1, total, self};
    array_append (&Trace.index, &t, sizeof(TraceIndex));
  }
  else
    t->calls++, t->total += total, t->self += self;
}

static void trace (const char * func, const char * file, int line)
{
  struct timeval tv;
  gettimeofday (&tv, NULL);
  if (Trace.t0 < 0)
    Trace.t0 = tv.tv_sec + tv.tv_usec/1e6;
  double t[2] = {(tv.tv_sec - Trace.t0) + tv.tv_usec/1e6, 0.};
  array_append (&Trace.stack, t, 2*sizeof(double));




}

static void end_trace (const char * func, const char * file, int line)
{
  struct timeval tv;
  gettimeofday (&tv, NULL);
  double te = (tv.tv_sec - Trace.t0) + tv.tv_usec/1e6;
  double * t = (double *) Trace.stack.p;
  assert (Trace.stack.len >= 2*sizeof(double));
  t += Trace.stack.len/sizeof(double) - 2;
  Trace.stack.len -= 2*sizeof(double);
  double dt = te - t[0];




  trace_add (func, file, line, dt, dt - t[1]);
  if (Trace.stack.len >= 2*sizeof(double)) {
    t -= 2;
    t[1] += dt;
  }
}

static int compar_self (const void * p1, const void * p2)
{
  const TraceIndex * t1 = p1, * t2 = p2;
  return t1->self < t2->self;
}

static void trace_off()
{
  int i, len = Trace.index.len/sizeof(TraceIndex);
  double total = 0.;
  TraceIndex * t;
  for (i = 0, t = (TraceIndex *) Trace.index.p; i < len; i++, t++)
    total += t->self;
  qsort (Trace.index.p, len, sizeof(TraceIndex), compar_self);
  fprintf (qstdout(), "   calls    total     self   %% total   function\n");
  for (i = 0, t = (TraceIndex *) Trace.index.p; i < len; i++, t++) {
    fprintf (qstdout(), "%8d   %6.2f   %6.2f     %4.1f%%   %s():%s:%d\n",
      t->calls, t->total, t->self, t->self*100./total,
      t->func, t->file, t->line);
    pfree (t->func,__func__,__FILE__,__LINE__); pfree (t->file,__func__,__FILE__,__LINE__);
  }

  pfree (Trace.index.p,__func__,__FILE__,__LINE__);
  Trace.index.p = NULL;
  Trace.index.len = Trace.index.max = 0;

  pfree (Trace.stack.p,__func__,__FILE__,__LINE__);
  Trace.stack.p = NULL;
  Trace.stack.len = Trace.stack.max = 0;
}

#else
# define trace(...)
# define end_trace(...)
#endif



#if _OPENMP

#include <omp.h>
#define OMP(x) _Pragma(#x)
#define tid() omp_get_thread_num()
#define pid() 0
#define npe() omp_get_num_threads()
#define mpi_all_reduce(v,type,op)
#define mpi_all_reduce_double(v,op)

#elif 1

#include <mpi.h>
#define OMP(x)

static bool in_prof = false;
static double prof_start, _prof;
#define prof_start(name)\
  assert (!in_prof); in_prof = true;\
  prof_start = MPI_Wtime();\

#line 590

#define prof_stop()\
  assert (in_prof); in_prof = false;\
  _prof = MPI_Wtime();\
  mpi_time += _prof - prof_start;\

#line 595


#if FAKE_MPI
#define mpi_all_reduce(v,type,op)
#define mpi_all_reduce_double(v,op)
#else

int mpi_all_reduce0 (void *sendbuf, void *recvbuf, int count,
       MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{ trace ("mpi_all_reduce0", "/home/miro3/Documents/Programming/basilisk/src/common.h", 604);
  { int _ret =  MPI_Allreduce (sendbuf, recvbuf, count, datatype, op, comm); end_trace("mpi_all_reduce0", "/home/miro3/Documents/Programming/basilisk/src/common.h", 605);  return _ret; }
 end_trace("mpi_all_reduce0", "/home/miro3/Documents/Programming/basilisk/src/common.h", 606); }
#define mpi_all_reduce(v,type,op) {\
  prof_start ("mpi_all_reduce");\
  union { int a; float b; double c;} global;\
  mpi_all_reduce0 (&(v), &global, 1, type, op, MPI_COMM_WORLD);\
  memcpy (&(v), &global, sizeof (v));\
  prof_stop();\
}\

#line 614

#define mpi_all_reduce_double(v,op) {\
  prof_start ("mpi_all_reduce");\
  double global, tmp = v;\
  mpi_all_reduce0 (&tmp, &global, 1, MPI_DOUBLE, op, MPI_COMM_WORLD);\
  v = global;\
  prof_stop();\
}\

#line 622


#endif

static int mpi_rank, mpi_npe;
#define tid() mpi_rank
#define pid() mpi_rank
#define npe() mpi_npe

#define QFILE FILE

static FILE * qstderr (void)
{
  static QFILE * fp = NULL;
  if (!fp) {
    if (mpi_rank > 0) {
      char name[80];
      sprintf (name, "log-%d", mpi_rank);
      fp = fopen (name, "w");
    }
    else
      fp = systderr;
  }
  return fp;
}

static FILE * qstdout (void)
{
  static QFILE * fp = NULL;
  if (!fp) {
    if (mpi_rank > 0) {
      char name[80];
      sprintf (name, "out-%d", mpi_rank);
      fp = fopen (name, "w");
    }
    else
      fp = systdout;
  }
  return fp;
}

static void finalize (void)
{
  MPI_Finalize();
}

void mpi_init()
{
  int initialized;
  MPI_Initialized (&initialized);
  if (!initialized) {
    MPI_Init (NULL, NULL);
    MPI_Comm_set_errhandler (MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    atexit (finalize);
    MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &mpi_npe);
    srand (mpi_rank + 1);
    if (mpi_rank > 0) {
      ferr = fopen ("/dev/null", "w");
      fout = fopen ("/dev/null", "w");
    }
    else {
      ferr = qstderr();
      fout = qstdout();
    }
    char * etrace = getenv ("MALLOC_TRACE"), name[80];
    if (etrace && mpi_rank > 0) {
      sprintf (name, "%s-%d", etrace, mpi_rank);
      setenv ("MALLOC_TRACE", name, 1);
    }
#if MTRACE == 1
    etrace = getenv ("MTRACE");
    if (!etrace)
      etrace = "mtrace";
    if (mpi_rank > 0) {
      sprintf (name, "%s-%d", etrace, mpi_rank);
      pmtrace.fp = fopen (name, "w");
      pmtrace.fname = systrdup(name);
    }
    else {
      pmtrace.fp = fopen (etrace, "w");
      pmtrace.fname = systrdup(etrace);
    }
#endif
  }
}

#else

#define OMP(x)
#define tid() 0
#define pid() 0
#define npe() 1
#define mpi_all_reduce(v,type,op)
#define mpi_all_reduce_double(v,op)

#endif

void init_solver()
{
#if _CADNA
  cadna_init (-1);
#endif
#if 1
  mpi_init();
#elif MTRACE == 1
  char * etrace = getenv ("MTRACE");
  pmtrace.fp = fopen (etrace ? etrace : "mtrace", "w");
  pmtrace.fname = systrdup (etrace ? etrace : "mtrace");
#endif
}

#define OMP_PARALLEL() OMP(omp parallel)

#define NOT_UNUSED(x) (void)(x)

#define VARIABLES ;
#define val(a,k,l,m) data(k,l,m)[a.i]

double _val_higher_dimension = 0.;
#define _val_higher_dimension(x,a,b,c) _val_higher_dimension
#line 751 "/home/miro3/Documents/Programming/basilisk/src/common.h"
#if (_GNU_SOURCE || __APPLE__) && !_OPENMP && !_CADNA
double undefined;
# if __APPLE__
# include <stdint.h>
# include "fp_osx.h"
# endif
# define enable_fpe(flags) feenableexcept (flags)
# define disable_fpe(flags) fedisableexcept (flags)
static void set_fpe (void) {
  int64_t lnan = 0x7ff0000000000001;
  assert (sizeof (int64_t) == sizeof (double));
  memcpy (&undefined, &lnan, sizeof (double));
  enable_fpe (FE_DIVBYZERO|FE_INVALID);
}
#else
# define undefined ((double) DBL_MAX)
# define enable_fpe(flags)
# define disable_fpe(flags)
static void set_fpe (void) {}
#endif


typedef struct {
  long n;
  long tn;
  int depth;
  int maxdepth;
} Grid;
Grid * grid = NULL;

double X0 = 0., Y0 = 0., Z0 = 0.;

double L0 = 1.;




int N = 16;


typedef struct { int i; } scalar;

typedef struct {
  scalar x;

  scalar y;


  scalar z;

} vector;

typedef struct {
  vector x;

  vector y;


  vector z;

} tensor;

typedef struct {
  double x, y, z;
} coord;
#line 828 "/home/miro3/Documents/Programming/basilisk/src/common.h"
void normalize (coord * n)
{
  double norm = 0.;
  {
#line 831

    norm += sq(n->x);
#line 831

    norm += sq(n->y);
#line 831

    norm += sq(n->z);}
  norm = sqrt(norm);
  {
#line 834

    n->x /= norm;
#line 834

    n->y /= norm;
#line 834

    n->z /= norm;}
}

struct _origin { double x, y, z; };

void origin (struct _origin p) {
  X0 = p.x; Y0 = p.y; Z0 = p.z;
}

void size (double L) {
  L0 = L;
}

double zero (double s0, double s1, double s2) { return 0.; }
#line 857 "/home/miro3/Documents/Programming/basilisk/src/common.h"
  enum { right, left, top, bottom, front, back };

int nboundary = 2*3;



#define dirichlet(x) (2.*(x) - val(_s,0,0,0))
#define dirichlet_homogeneous() (- val(_s,0,0,0))
#define neumann(x) (Delta*(x) + val(_s,0,0,0))
#define neumann_homogeneous() (val(_s,0,0,0))

double * _constant = NULL;
extern size_t datasize;
typedef struct _Point Point;

#line 1 "/home/miro3/Documents/Programming/basilisk/src/grid/boundaries.h"


typedef struct _Boundary Boundary;

struct _Boundary {
  void (* destroy) (Boundary * b);
  void (* level) (const Boundary * b, scalar * list, int l);

  void (* restriction) (const Boundary * b, scalar * list, int l);
};

static Boundary ** boundaries = NULL;

void add_boundary (Boundary * b) {
  int len = 0;
  if (boundaries) {
    Boundary ** i = boundaries;
    while (*i++) len++;
  }
  boundaries = (Boundary * *) prealloc (boundaries, (len + 2)*sizeof(Boundary *),__func__,__FILE__,__LINE__);
  boundaries[len] = b;
  boundaries[len+1] = NULL;
}

void free_boundaries() {
  if (!boundaries)
    return;
  Boundary ** i = boundaries, * b;
  while ((b = *i++))
    if (b->destroy)
      b->destroy (b);
    else
      pfree (b,__func__,__FILE__,__LINE__);
  pfree (boundaries,__func__,__FILE__,__LINE__);
  boundaries = NULL;
}
#line 47 "/home/miro3/Documents/Programming/basilisk/src/grid/boundaries.h"
typedef struct {
  Boundary parent;
  int d;
} BoxBoundary;
#line 873 "/home/miro3/Documents/Programming/basilisk/src/common.h"



typedef struct {

#line 878 "/home/miro3/Documents/Programming/basilisk/src/common.h"

  double (** boundary) (Point, Point, scalar);
  double (** boundary_homogeneous) (Point, Point, scalar);
  double (* gradient) (double, double, double);
  void (* delete) (scalar);
  char * name;
  struct {
    int x;

    int y;


    int z;

  } d;
  vector v;
  bool face, nodump;

#line 17 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"

  void (* prolongation) (Point, scalar);
  void (* restriction) (Point, scalar);

#line 8 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"

  void (* refine) (Point, scalar);

#line 87 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"

  void (* coarsen) (Point, scalar);

} _Attributes;
_Attributes * _attribute;
#line 876 "/home/miro3/Documents/Programming/basilisk/src/common.h"























int list_len (scalar * list)
{
  if (!list) return 0;
  int ns = 0;
  if (list) for (scalar s = *list, *_i0 = list; ((scalar *)&s)->i >= 0; s = *++_i0) ns++;
  return ns;
}

scalar * list_append (scalar * list, scalar s)
{
  int len = list_len (list);
  list = (scalar *) prealloc (list, (len + 2)*sizeof(scalar),__func__,__FILE__,__LINE__);
  list[len] = s;
  list[len + 1].i = -1;
  return list;
}

scalar * list_add (scalar * list, scalar s)
{
  if (list) for (scalar t = *list, *_i1 = list; ((scalar *)&t)->i >= 0; t = *++_i1)
    if (t.i == s.i)
      return list;
  return list_append (list, s);
}

int list_lookup (scalar * l, scalar s)
{
  if (l != NULL)
    if (l) for (scalar s1 = *l, *_i2 = l; ((scalar *)&s1)->i >= 0; s1 = *++_i2)
      if (s1.i == s.i)
 return true;
  return false;
}

scalar * list_copy (scalar * l)
{
  scalar * list = NULL;
  if (l != NULL)
    if (l) for (scalar s = *l, *_i3 = l; ((scalar *)&s)->i >= 0; s = *++_i3)
      list = list_append (list, s);
  return list;
}

scalar * list_concat (scalar * l1, scalar * l2)
{
  scalar * l3 = list_copy (l1);
  if (l2) for (scalar s = *l2, *_i4 = l2; ((scalar *)&s)->i >= 0; s = *++_i4)
    l3 = list_append (l3, s);
  return l3;
}

void list_print (scalar * l, FILE * fp)
{
  int i = 0;
  if (l) for (scalar s = *l, *_i5 = l; ((scalar *)&s)->i >= 0; s = *++_i5)
    fprintf (fp, "%s%s", i++ == 0 ? "{" : ",", _attribute[s.i].name);
  fputs (i > 0 ? "}\n" : "{}\n", fp);
}

int vectors_len (vector * list)
{
  if (!list) return 0;
  int nv = 0;
  if (list) for (vector v = *list, *_i6 = list; ((scalar *)&v)->i >= 0; v = *++_i6) nv++;
  return nv;
}

vector * vectors_append (vector * list, vector v)
{
  int len = vectors_len (list);
  list = (vector *) prealloc (list, (len + 2)*sizeof(vector),__func__,__FILE__,__LINE__);
  list[len] = v;
  list[len + 1] = (vector){{-1}};
  return list;
}

vector * vectors_add (vector * list, vector v)
{
  if (list) for (vector w = *list, *_i7 = list; ((scalar *)&w)->i >= 0; w = *++_i7) {
    bool id = true;
    {
#line 979

      if (w.x.i != v.x.i)
 id = false;
#line 979

      if (w.y.i != v.y.i)
 id = false;
#line 979

      if (w.z.i != v.z.i)
 id = false;}
    if (id)
      return list;
  }
  return vectors_append (list, v);
}

vector * vectors_copy (vector * l)
{
  vector * list = NULL;
  if (l != NULL)
    if (l) for (vector v = *l, *_i8 = l; ((scalar *)&v)->i >= 0; v = *++_i8)
      list = vectors_append (list, v);
  return list;
}

vector * vectors_from_scalars (scalar * s)
{
  vector * list = NULL;
  while (s->i >= 0) {
    vector v;
    {
#line 1002
 {
      assert (s->i >= 0);
      v.x = *s++;
    }
#line 1002
 {
      assert (s->i >= 0);
      v.y = *s++;
    }
#line 1002
 {
      assert (s->i >= 0);
      v.z = *s++;
    }}
    list = vectors_append (list, v);
  }
  return list;
}

int tensors_len (tensor * list)
{
  if (!list) return 0;
  int nt = 0;
  if (list) for (tensor t = *list, *_i9 = list; ((scalar *)&t)->i >= 0; t = *++_i9) nt++;
  return nt;
}

tensor * tensors_append (tensor * list, tensor t)
{
  int len = tensors_len (list);
  list = (tensor *) prealloc (list, (len + 2)*sizeof(tensor),__func__,__FILE__,__LINE__);
  list[len] = t;
  list[len + 1] = (tensor){{{-1}}};
  return list;
}

tensor * tensors_from_vectors (vector * v)
{
  tensor * list = NULL;
  while (v->x.i >= 0) {
    tensor t;
    {
#line 1033
 {
      assert (v->x.i >= 0);
      t.x = *v++;
    }
#line 1033
 {
      assert (v->y.i >= 0);
      t.y = *v++;
    }
#line 1033
 {
      assert (v->z.i >= 0);
      t.z = *v++;
    }}
    list = tensors_append (list, t);
  }
  return list;
}

scalar * all = NULL;



scalar (* init_scalar) (scalar, const char *);
vector (* init_vector) (vector, const char *);
tensor (* init_tensor) (tensor, const char *);
vector (* init_face_vector) (vector, const char *);





typedef struct _Event Event;
typedef int (* Expr) (int *, double *, Event *);

struct _Event {
  int last, nexpr;
  int (* action) (const int, const double, Event *);
  Expr expr[3];
  int * arrayi;
  double * arrayt;
  char * file;
  int line;
  char * name;
  double t;
  int i, a;
  void * data;
  Event * next;
};

static Event * Events = NULL;

int iter = 0, inext = 0;
double t = 0, tnext = 0;
void init_events (void);
void event_register (Event event);
void _init_solver (void);



#if 1
static double mpi_time = 0.;
#endif

typedef struct {
  clock_t c;
  struct timeval tv;
  double tm;
} timer;

timer timer_start (void)
{
  timer t;
  t.c = clock();
  gettimeofday (&t.tv, NULL);
#if 1
  t.tm = mpi_time;
#endif
  return t;
}

double timer_elapsed (timer t)
{
  struct timeval tvend;
  gettimeofday (&tvend, NULL);
  return ((tvend.tv_sec - t.tv.tv_sec) +
   (tvend.tv_usec - t.tv.tv_usec)/1e6);
}



vector zerof= {{_NVARMAX + 0},{_NVARMAX + 1},{_NVARMAX + 2}};
vector unityf= {{_NVARMAX + 3},{_NVARMAX + 4},{_NVARMAX + 5}};
scalar unity= {_NVARMAX + 6};
scalar zeroc= {_NVARMAX + 7};



 vector fm = {{_NVARMAX + 3},{_NVARMAX + 4},{_NVARMAX + 5}};
 scalar cm = {(_NVARMAX + 6)};



static FILE ** qpopen_pipes = NULL;

FILE * qpopen (const char * command, const char * type)
{
  FILE * fp = popen (command, type);
  if (fp) {
    FILE ** i = qpopen_pipes;
    int n = 0;
    while (i && *i) { n++; i++; }
    qpopen_pipes = (FILE * *) prealloc (qpopen_pipes, (n + 2)*sizeof(FILE *),__func__,__FILE__,__LINE__);
    qpopen_pipes[n] = fp;
    qpopen_pipes[n+1] = NULL;
  }
  return fp;
}

int qpclose (FILE * fp)
{
  FILE ** i = qpopen_pipes;
  while (i && *i) {
    if (*i == fp)
      *i = (FILE *) 1;
    i++;
  }
  return pclose (fp);
}

static void qpclose_all()
{
  FILE ** i = qpopen_pipes;
  while (i && *i) {
    if (*i != (FILE *) 1)
      pclose (*i);
    i++;
  }
  pfree (qpopen_pipes,__func__,__FILE__,__LINE__);
  qpopen_pipes = NULL;
}






FILE * lfopen (const char * name, const char * mode)
{
  char fname[80];
  sprintf (fname, "%s-%d", name, pid());
  return fopen (fname, mode);
}



void * matrix_new (int n, int p, size_t size)
{
  void ** m = ((void * *) pmalloc ((n)*sizeof(void *),__func__,__FILE__,__LINE__));
  char * a = ((char *) pmalloc ((n*p*size)*sizeof(char),__func__,__FILE__,__LINE__));
  for (int i = 0; i < n; i++)
    m[i] = a + i*p*size;
  return m;
}

double matrix_inverse (double ** m, int n, double pivmin)
{
  int indxc[n], indxr[n], ipiv[n];
  int i, icol = 0, irow = 0, j, k, l, ll;
  double big, dum, pivinv, minpiv = HUGE;

  for (j = 0; j < n; j++)
    ipiv[j] = -1;

  for (i = 0; i < n; i++) {
    big = 0.0;
    for (j = 0; j < n; j++)
      if (ipiv[j] != 0)
 for (k = 0; k < n; k++) {
   if (ipiv[k] == -1) {
     if (fabs (m[j][k]) >= big) {
       big = fabs (m[j][k]);
       irow = j;
       icol = k;
     }
   }
 }
    ipiv[icol]++;
    if (irow != icol)
      for (l = 0; l < n; l++)
 swap (double, m[irow][l], m[icol][l]);
    indxr[i] = irow;
    indxc[i] = icol;
    if (fabs (m[icol][icol]) <= pivmin)
      return 0.;
    if (fabs (m[icol][icol]) < minpiv)
      minpiv = fabs (m[icol][icol]);
    pivinv = 1.0/m[icol][icol];
    m[icol][icol] = 1.0;
    for (l = 0; l < n; l++) m[icol][l] *= pivinv;
    for (ll = 0; ll < n; ll++)
      if (ll != icol) {
 dum = m[ll][icol];
 m[ll][icol] = 0.0;
 for (l = 0; l < n; l++)
   m[ll][l] -= m[icol][l]*dum;
      }
  }
  for (l = n - 1; l >= 0; l--) {
    if (indxr[l] != indxc[l])
      for (k = 0; k < n; k++)
 swap (double, m[k][indxr[l]], m[k][indxc[l]]);
  }
  return minpiv;
}

void matrix_free (void * m)
{
  pfree (((void **) m)[0],__func__,__FILE__,__LINE__);
  pfree (m,__func__,__FILE__,__LINE__);
}
#line 13 "probing-cpp.c"
#line 1 "probing.c"
#line 1 "probing.h"
#line 1 "./probing.h"



#line 1 "grid/octree.h"
#line 1 "/home/miro3/Documents/Programming/basilisk/src/grid/octree.h"


#line 1 "grid/tree.h"
#line 1 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
#line 1 "grid/mempool.h"
#line 1 "/home/miro3/Documents/Programming/basilisk/src/grid/mempool.h"





typedef struct _Pool Pool;

struct _Pool {
  Pool * next;
};

typedef struct {
  char * first, * lastb;
  size_t size;
  size_t poolsize;
  Pool * pool, * last;
} Mempool;

typedef struct {
  char * next;
} FreeBlock;

Mempool * mempool_new (size_t poolsize, size_t size)
{

  assert (poolsize % 8 == 0);
  assert (size >= sizeof(FreeBlock));


  poolsize = min(1 << 20, poolsize + sizeof(Pool));
  Mempool * m = ((Mempool *) pcalloc (1, sizeof(Mempool),__func__,__FILE__,__LINE__));
  m->poolsize = poolsize;
  m->size = size;
  return m;
}

void mempool_destroy (Mempool * m)
{
  Pool * p = m->pool;
  while (p) {
    Pool * next = p->next;
    pfree (p,__func__,__FILE__,__LINE__);
    p = next;
  }
  pfree (m,__func__,__FILE__,__LINE__);
}

void * mempool_alloc (Mempool * m)
{
  if (!m->first) {

    Pool * p = (Pool *) pmalloc (m->poolsize,__func__,__FILE__,__LINE__);
    p->next = NULL;
    if (m->last)
      m->last->next = p;
    else
      m->pool = p;
    m->last = p;
    m->first = m->lastb = ((char *)m->last) + sizeof(Pool);
    FreeBlock * b = (FreeBlock *) m->first;
    b->next = NULL;
  }
  void * ret = m->first;
  FreeBlock * b = (FreeBlock *) ret;
  char * next = b->next;
  if (!next) {
    m->lastb += m->size;
    next = m->lastb;
    if (next + m->size > ((char *) m->last) + m->poolsize)
      next = NULL;
    else {
      FreeBlock * b = (FreeBlock *) next;
      b->next = NULL;
    }
  }
  m->first = next;
#if TRASH
  double * v = (double *) ret;
  for (int i = 0; i < m->size/sizeof(double); i++)
    v[i] = undefined;
#endif
  return ret;
}

void * mempool_alloc0 (Mempool * m)
{
  void * ret = mempool_alloc (m);
  memset (ret, 0, m->size);
  return ret;
}

void mempool_free (Mempool * m, void * p)
{
#if TRASH
  double * v = (double *) p;
  for (int i = 0; i < m->size/sizeof(double); i++)
    v[i] = undefined;
#endif
  FreeBlock * b = (FreeBlock *) p;
  b->next = m->first;
  m->first = (char *) p;
}
#line 2 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
#line 10 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
# define BGHOSTS 1
#line 22 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
typedef struct {
  unsigned short flags;

  unsigned short neighbors;
  int pid;
} Cell;

enum {
  active = 1 << 0,
  leaf = 1 << 1,
  border = 1 << 2,
  vertex = 1 << 3,
  user = 4,

  face_x = 1 << 0

  , face_y = 1 << 1


  , face_z = 1 << 2

};

#define is_active(cell) ((cell).flags & active)
#define is_leaf(cell) ((cell).flags & leaf)
#define is_coarse() ((cell).neighbors > 0)
#define is_border(cell) ((cell).flags & border)
#define is_local(cell) ((cell).pid == pid())



typedef struct {
  int i;

  int j;


  int k;

} IndexLevel;

typedef struct {
  IndexLevel * p;
  int n, nm;
} CacheLevel;

typedef struct {
  int i;

  int j;


  int k;

  int level, flags;
} Index;

typedef struct {
  Index * p;
  int n, nm;
} Cache;




static void * new_refarray (size_t len, size_t size) {
  return pcalloc (len + 1, size,__func__,__FILE__,__LINE__);
}

static void refarray (void * p, size_t len, size_t size) {
  int * refcount = (int *)(((char *)p) + len*size);
  (*refcount)++;
}

static bool unrefarray (void * p, size_t len, size_t size) {
  int * refcount = (int *)(((char *)p) + len*size);
  (*refcount)--;
  if (*refcount == 0) {
    pfree (p,__func__,__FILE__,__LINE__);
    return true;
  }
  return false;
}




typedef struct {





  char **** m;

  Mempool * pool;
  int nc;
  int len;
} Layer;

static size_t _size (size_t depth)
{
  return (1 << depth) + 2*2;
}

static size_t poolsize (size_t depth, size_t size)
{






  return cube(_size(depth))*size;

}

static Layer * new_layer (int depth)
{
  Layer * l = ((Layer *) pmalloc ((1)*sizeof(Layer),__func__,__FILE__,__LINE__));
  l->len = _size (depth);
  if (depth == 0)
    l->pool = NULL;
  else {
    size_t size = sizeof(Cell) + datasize;


    l->pool = mempool_new (poolsize (depth, size), (1 << 3)*size);
  }





  l->m = ((char *** *) pcalloc (l->len, sizeof(char ***),__func__,__FILE__,__LINE__));

  l->nc = 0;
  return l;
}

static void destroy_layer (Layer * l)
{
  if (l->pool)
    mempool_destroy (l->pool);
  pfree (l->m,__func__,__FILE__,__LINE__);
  pfree (l,__func__,__FILE__,__LINE__);
}
#line 186 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
static void layer_add_row (Layer * l, int i, int j)
{







  if (!l->m[i]) {
    l->m[i] = (char ***) new_refarray (l->len, sizeof (char **));
    l->nc++;
  }
  refarray (l->m[i], l->len, sizeof(char *));
  if (!l->m[i][j])
    l->m[i][j] = (char **) new_refarray (l->len, sizeof (char *));
  refarray (l->m[i][j], l->len, sizeof(char *));

}

static bool layer_remove_row (Layer * l, int i, int j)
{

  if (unrefarray (l->m[i][j], l->len, sizeof (char *)))
    l->m[i][j] = NULL;

  if (unrefarray (l->m[i], l->len, sizeof (char *))) {
    l->m[i] = NULL;
    if (--l->nc == 0) {
      destroy_layer (l);
      return true;
    }
    assert (l->nc >= 0);
  }
  return false;
}




typedef struct {
  Grid g;
  Layer ** L;

  Cache leaves;
  Cache faces;
  Cache vertices;
  Cache refined;
  CacheLevel * active;
  CacheLevel * prolongation;
  CacheLevel * boundary;

  CacheLevel * restriction;

  bool dirty;
} Tree;



struct _Point {

  int i;

  int j;


  int k;

  int level;
};
static Point last_point;



static void cache_level_append (CacheLevel * c, Point p)
{
  if (c->n >= c->nm) {
    c->nm += 128;
    c->p = (IndexLevel *) prealloc (c->p, (c->nm)*sizeof(IndexLevel),__func__,__FILE__,__LINE__);
  }
  c->p[c->n].i = p.i;

  c->p[c->n].j = p.j;


  c->p[c->n].k = p.k;

  c->n++;
}

static void cache_level_shrink (CacheLevel * c)
{
  if (c->nm > (c->n/128 + 1)*128) {
    c->nm = (c->n/128 + 1)*128;
    assert (c->nm > c->n);
    c->p = (IndexLevel *) prealloc (c->p, sizeof (Index)*c->nm,__func__,__FILE__,__LINE__);
  }
}

static void cache_append (Cache * c, Point p, unsigned short flags)
{
  if (c->n >= c->nm) {
    c->nm += 128;
    c->p = (Index *) prealloc (c->p, (c->nm)*sizeof(Index),__func__,__FILE__,__LINE__);
  }
  c->p[c->n].i = p.i;

  c->p[c->n].j = p.j;


  c->p[c->n].k = p.k;

  c->p[c->n].level = p.level;
  c->p[c->n].flags = flags;
  c->n++;
}

void cache_shrink (Cache * c)
{
  cache_level_shrink ((CacheLevel *)c);
}
#line 362 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
#define allocated(a,l,n) (point.i+a >= 0 &&\
         point.i+a < (1 << point.level) + 2*2 &&\
         ((Tree *)grid)->L[point.level]->m[point.i+a] &&\
         point.j+l >= 0 &&\
         point.j+l < (1 << point.level) + 2*2 &&\
         ((Tree *)grid)->L[point.level]->m[point.i+a][point.j+l] &&\
         point.k+n >= 0 &&\
         point.k+n < (1 << point.level) + 2*2 &&\
         ((Tree *)grid)->L[point.level]->m[point.i+a][point.j+l]\
         [point.k+n])\

#line 372


#define NEIGHBOR(a,l,n) (((Tree *)grid)->L[point.level]->m[point.i+a][point.j+l]\
                                          [point.k+n])\

#line 376

#define PARENT(a,l,n) (((Tree *)grid)->L[point.level-1]->m[(point.i+2)/2+a]\
    [(point.j+2)/2+l][(point.k+2)/2+n])\

#line 379

#define allocated_child(a,l,n) (level < depth() &&\
         point.i > 0 && point.i <= (1 << level) + 2 &&\
         point.j > 0 && point.j <= (1 << level) + 2 &&\
         point.k > 0 && point.k <= (1 << level) + 2 &&\
         ((Tree *)grid)->L[point.level+1]->m[2*point.i-2 +a]\
   && ((Tree *)grid)->L[point.level+1]->m[2*point.i-2 +a][2*point.j-2 +l]\
   && ((Tree *)grid)->L[point.level+1]->m[2*point.i-2 +a][2*point.j-2 +l]\
         [2*point.k-2 +n])\

#line 388

#define CHILD(a,l,n) (((Tree *)grid)->L[point.level+1]->m[2*point.i-2 +a]\
      [2*point.j-2 +l][2*point.k-2 +n])\

#line 391


#define CELL(m) (*((Cell *)(m)))


#define depth() (grid->depth)
#define aparent(k,l,n) CELL(PARENT(k,l,n))
#define child(k,l,n) CELL(CHILD(k,l,n))


#define cell CELL(NEIGHBOR(0,0,0))
#define neighbor(k,l,n) CELL(NEIGHBOR(k,l,n))
#define neighborp(l,m,n) (Point) {\
    point.i + l,\
\
    point.j + m,\
\
\
    point.k + n,\
\
    point.level }\

#line 412



#define data(k,l,n) ((double *) (NEIGHBOR(k,l,n) + sizeof(Cell)))
#define fine(a,k,l,n) ((double *) (CHILD(k,l,n) + sizeof(Cell)))[a.i]
#define coarse(a,k,l,n) ((double *) (PARENT(k,l,n) + sizeof(Cell)))[a.i]

#define POINT_VARIABLES\
  VARIABLES\
  int level = point.level; NOT_UNUSED(level);\
\
\
\
\
\
\
\
  struct { int x, y, z; } child = {\
    2*((point.i+2)%2)-1, 2*((point.j+2)%2)-1, 2*((point.k+2)%2)-1\
  };\
\
  NOT_UNUSED(child);\
  Point parent = point; NOT_UNUSED(parent);\
  parent.level--;\
  parent.i = (point.i + 2)/2;\
\
  parent.j = (point.j + 2)/2;\
\
\
  parent.k = (point.k + 2)/2;\
\

#line 443


#line 1 "grid/foreach_cell.h"
#line 1 "/home/miro3/Documents/Programming/basilisk/src/grid/foreach_cell.h"
#line 66 "/home/miro3/Documents/Programming/basilisk/src/grid/foreach_cell.h"
#define foreach_cell_root(root)\
  {\
    int ig = 0, jg = 0; NOT_UNUSED(ig); NOT_UNUSED(jg);\
    Point point;\
\
\
\
\
\
    int kg = 0; NOT_UNUSED(kg);\
    struct { int l, i, j, k, stage; } stack[20];\
\
    int _s = -1;\
    { _s++; stack[_s].l = 0; stack[_s].i = root.i; stack[_s].j = root.j; stack[_s].k = root.k; stack[_s].stage = 0; };\
    while (_s >= 0) {\
      int stage;\
      { point.level = stack[_s].l; point.i = stack[_s].i; point.j = stack[_s].j; point.k = stack[_s].k; stage = stack[_s].stage; _s--; };\
      if (!allocated (0,0,0))\
 continue;\
      switch (stage) {\
      case 0: {\
 POINT_VARIABLES;\
\

#line 89

#define end_foreach_cell_root()\
        if (point.level < grid->depth) {\
   { _s++; stack[_s].l = point.level; stack[_s].i = point.i; stack[_s].j = point.j; stack[_s].k = point.k; stack[_s].stage = 1; };\
          { _s++; stack[_s].l = point.level + 1; stack[_s].i = (2*point.i - 2); stack[_s].j = (2*point.j - 2); stack[_s].k = (2*point.k - 2); stack[_s].stage = 0; };\
        }\
        break;\
      }\
\
      case 1: { _s++; stack[_s].l = point.level; stack[_s].i = point.i; stack[_s].j = point.j; stack[_s].k = point.k; stack[_s].stage = 2; };\
       { _s++; stack[_s].l = point.level + 1; stack[_s].i = (2*point.i - 2); stack[_s].j = (2*point.j - 2); stack[_s].k = ((2*point.k - 2) + 1); stack[_s].stage = 0; }; break;\
      case 2: { _s++; stack[_s].l = point.level; stack[_s].i = point.i; stack[_s].j = point.j; stack[_s].k = point.k; stack[_s].stage = 3; };\
       { _s++; stack[_s].l = point.level + 1; stack[_s].i = (2*point.i - 2); stack[_s].j = ((2*point.j - 2) + 1); stack[_s].k = (2*point.k - 2); stack[_s].stage = 0; }; break;\
      case 3: { _s++; stack[_s].l = point.level; stack[_s].i = point.i; stack[_s].j = point.j; stack[_s].k = point.k; stack[_s].stage = 4; };\
       { _s++; stack[_s].l = point.level + 1; stack[_s].i = (2*point.i - 2); stack[_s].j = ((2*point.j - 2) + 1); stack[_s].k = ((2*point.k - 2) + 1); stack[_s].stage = 0; }; break;\
      case 4: { _s++; stack[_s].l = point.level; stack[_s].i = point.i; stack[_s].j = point.j; stack[_s].k = point.k; stack[_s].stage = 5; };\
       { _s++; stack[_s].l = point.level + 1; stack[_s].i = ((2*point.i - 2) + 1); stack[_s].j = (2*point.j - 2); stack[_s].k = (2*point.k - 2); stack[_s].stage = 0; }; break;\
      case 5: { _s++; stack[_s].l = point.level; stack[_s].i = point.i; stack[_s].j = point.j; stack[_s].k = point.k; stack[_s].stage = 6; };\
       { _s++; stack[_s].l = point.level + 1; stack[_s].i = ((2*point.i - 2) + 1); stack[_s].j = (2*point.j - 2); stack[_s].k = ((2*point.k - 2) + 1); stack[_s].stage = 0; }; break;\
      case 6: { _s++; stack[_s].l = point.level; stack[_s].i = point.i; stack[_s].j = point.j; stack[_s].k = point.k; stack[_s].stage = 7; };\
       { _s++; stack[_s].l = point.level + 1; stack[_s].i = ((2*point.i - 2) + 1); stack[_s].j = ((2*point.j - 2) + 1); stack[_s].k = (2*point.k - 2); stack[_s].stage = 0; }; break;\
      case 7: { _s++; stack[_s].l = point.level + 1; stack[_s].i = ((2*point.i - 2) + 1); stack[_s].j = ((2*point.j - 2) + 1); stack[_s].k = ((2*point.k - 2) + 1); stack[_s].stage = 0; }; break;\
\
      }\
    }\
  }\

#line 123


#define foreach_cell() {\
\
\
\
\
\
  Point root = {2,2,2,0};\
\
  foreach_cell_root (root)\

#line 134

#define end_foreach_cell() end_foreach_cell_root() }

#define foreach_cell_all() {\
  Point root = { .level = 0 };\
  for (root.i = 0; root.i <= 2*2; root.i++)\
\
    for (root.j = 0; root.j <= 2*2; root.j++)\
\
\
      for (root.k = 0; root.k <= 2*2; root.k++)\
\
 foreach_cell_root (root)\

#line 147

#define end_foreach_cell_all() end_foreach_cell_root() }

#define foreach_cell_post_root(condition, root)\
  {\
    int ig = 0, jg = 0; NOT_UNUSED(ig); NOT_UNUSED(jg);\
    Point point;\
\
\
\
\
\
    int kg = 0; NOT_UNUSED(kg);\
    struct { int l, i, j, k, stage; } stack[20];\
\
    int _s = -1;\
    { _s++; stack[_s].l = 0; stack[_s].i = root.i; stack[_s].j = root.j; stack[_s].k = root.k; stack[_s].stage = 0; };\
    while (_s >= 0) {\
      int stage;\
      { point.level = stack[_s].l; point.i = stack[_s].i; point.j = stack[_s].j; point.k = stack[_s].k; stage = stack[_s].stage; _s--; };\
      if (!allocated (0,0,0))\
 continue;\
      switch (stage) {\
      case 0: {\
        POINT_VARIABLES;\
 if (point.level == grid->depth) {\
   { _s++; stack[_s].l = point.level; stack[_s].i = point.i; stack[_s].j = point.j; stack[_s].k = point.k; stack[_s].stage = 8; };\
 }\
 else {\
   { _s++; stack[_s].l = point.level; stack[_s].i = point.i; stack[_s].j = point.j; stack[_s].k = point.k; stack[_s].stage = 1; };\
   if (condition)\
     { _s++; stack[_s].l = point.level + 1; stack[_s].i = (2*point.i - 2); stack[_s].j = (2*point.j - 2); stack[_s].k = (2*point.k - 2); stack[_s].stage = 0; };\
 }\
 break;\
      }\
\
      case 1:\
 { _s++; stack[_s].l = point.level; stack[_s].i = point.i; stack[_s].j = point.j; stack[_s].k = point.k; stack[_s].stage = 2; };\
 if (condition)\
   { _s++; stack[_s].l = point.level + 1; stack[_s].i = (2*point.i - 2); stack[_s].j = (2*point.j - 2); stack[_s].k = ((2*point.k - 2) + 1); stack[_s].stage = 0; };\
 break;\
      case 2:\
 { _s++; stack[_s].l = point.level; stack[_s].i = point.i; stack[_s].j = point.j; stack[_s].k = point.k; stack[_s].stage = 3; };\
 if (condition)\
   { _s++; stack[_s].l = point.level + 1; stack[_s].i = (2*point.i - 2); stack[_s].j = ((2*point.j - 2) + 1); stack[_s].k = (2*point.k - 2); stack[_s].stage = 0; };\
 break;\
      case 3:\
 { _s++; stack[_s].l = point.level; stack[_s].i = point.i; stack[_s].j = point.j; stack[_s].k = point.k; stack[_s].stage = 4; };\
 if (condition)\
   { _s++; stack[_s].l = point.level + 1; stack[_s].i = (2*point.i - 2); stack[_s].j = ((2*point.j - 2) + 1); stack[_s].k = ((2*point.k - 2) + 1); stack[_s].stage = 0; };\
 break;\
      case 4:\
 { _s++; stack[_s].l = point.level; stack[_s].i = point.i; stack[_s].j = point.j; stack[_s].k = point.k; stack[_s].stage = 5; };\
 if (condition)\
   { _s++; stack[_s].l = point.level + 1; stack[_s].i = ((2*point.i - 2) + 1); stack[_s].j = (2*point.j - 2); stack[_s].k = (2*point.k - 2); stack[_s].stage = 0; };\
 break;\
      case 5:\
 { _s++; stack[_s].l = point.level; stack[_s].i = point.i; stack[_s].j = point.j; stack[_s].k = point.k; stack[_s].stage = 6; };\
 if (condition)\
   { _s++; stack[_s].l = point.level + 1; stack[_s].i = ((2*point.i - 2) + 1); stack[_s].j = (2*point.j - 2); stack[_s].k = ((2*point.k - 2) + 1); stack[_s].stage = 0; };\
 break;\
      case 6:\
 { _s++; stack[_s].l = point.level; stack[_s].i = point.i; stack[_s].j = point.j; stack[_s].k = point.k; stack[_s].stage = 7; };\
 if (condition)\
   { _s++; stack[_s].l = point.level + 1; stack[_s].i = ((2*point.i - 2) + 1); stack[_s].j = ((2*point.j - 2) + 1); stack[_s].k = (2*point.k - 2); stack[_s].stage = 0; };\
 break;\
      case 7:\
 { _s++; stack[_s].l = point.level; stack[_s].i = point.i; stack[_s].j = point.j; stack[_s].k = point.k; stack[_s].stage = 8; };\
 if (condition)\
   { _s++; stack[_s].l = point.level + 1; stack[_s].i = ((2*point.i - 2) + 1); stack[_s].j = ((2*point.j - 2) + 1); stack[_s].k = ((2*point.k - 2) + 1); stack[_s].stage = 0; };\
 break;\
\
      default: {\
        POINT_VARIABLES;\
\

#line 244

#define end_foreach_cell_post_root()\
      }\
      }\
    }\
  }\

#line 250


#define foreach_cell_post(condition)\
  {\
\
\
\
\
\
    Point root = {2,2,2,0};\
\
    foreach_cell_post_root(condition, root)\

#line 262

#define end_foreach_cell_post() end_foreach_cell_post_root() }

#define foreach_cell_post_all(condition) {\
  Point root = { .level = 0 };\
  for (root.i = 0; root.i <= 2*2; root.i++)\
\
    for (root.j = 0; root.j <= 2*2; root.j++)\
\
\
      for (root.k = 0; root.k <= 2*2; root.k++)\
\
 foreach_cell_post_root (condition, root)\

#line 275

#define end_foreach_cell_post_all() end_foreach_cell_post_root() }

#define foreach_leaf() foreach_cell()\
  if (is_leaf (cell)) {\
    if (is_active(cell) && is_local(cell)) {\

#line 281

#define end_foreach_leaf() } continue; } end_foreach_cell()
#line 446 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
#line 481 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
#define foreach_child() {\
  int _i = 2*point.i - 2, _j = 2*point.j - 2, _k = 2*point.k - 2;\
  point.level++;\
  for (int _l = 0; _l < 2; _l++) {\
    point.i = _i + _l;\
    for (int _m = 0; _m < 2; _m++) {\
      point.j = _j + _m;\
      for (int _n = 0; _n < 2; _n++) {\
 point.k = _k + _n;\
 POINT_VARIABLES;\

#line 491

#define end_foreach_child()\
      }\
    }\
  }\
  point.i = (_i + 2)/2;point.j = (_j + 2)/2;point.k = (_k + 2)/2;\
  point.level--;\
}\

#line 499

#define foreach_child_break() _l = _m = _n = 2
#line 509 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
#define foreach_cache(_cache) {\
  OMP_PARALLEL() {\
  int ig = 0, jg = 0, kg = 0; NOT_UNUSED(ig); NOT_UNUSED(jg); NOT_UNUSED(kg);\
\
\
\
\
\
  Point point = {2,2,2,0};\
\
  int _k; unsigned short _flags; NOT_UNUSED(_flags);\
  OMP(omp for schedule(static))\
  for (_k = 0; _k < _cache.n; _k++) {\
    point.i = _cache.p[_k].i;\
\
    point.j = _cache.p[_k].j;\
\
\
    point.k = _cache.p[_k].k;\
\
    point.level = _cache.p[_k].level;\
    _flags = _cache.p[_k].flags;\
    POINT_VARIABLES;\

#line 532

#define end_foreach_cache() } } }

#define foreach_cache_level(_cache,_l) {\
  OMP_PARALLEL() {\
  int ig = 0, jg = 0, kg = 0; NOT_UNUSED(ig); NOT_UNUSED(jg); NOT_UNUSED(kg);\
\
\
\
\
\
  Point point = {2,2,2,0};\
\
  point.level = _l;\
  int _k;\
  OMP(omp for schedule(static))\
  for (_k = 0; _k < _cache.n; _k++) {\
    point.i = _cache.p[_k].i;\
\
    point.j = _cache.p[_k].j;\
\
\
    point.k = _cache.p[_k].k;\
\
    POINT_VARIABLES;\

#line 557

#define end_foreach_cache_level() } } }

#define foreach_boundary_level(_l) {\
  if (_l <= depth()) {\
    { if (((Tree *)grid)->dirty) update_cache_f(); };\
    CacheLevel _boundary = ((Tree *)grid)->boundary[_l];\
    foreach_cache_level (_boundary,_l)\

#line 565

#define end_foreach_boundary_level() end_foreach_cache_level(); }}



#define foreach_boundary(_b) {\
  for (int _l = depth(); _l >= 0; _l--)\
    foreach_boundary_level(_l) {\
      if ((- cell.pid - 1) == _b)\
 for (int _d = 0; _d < 3; _d++) {\
   for (int _i = -1; _i <= 1; _i += 2) {\
     if (_d == 0) ig = _i; else if (_d == 1) jg = _i; else kg = _i;\
     if (allocated(-ig,-jg,-kg) &&\
  is_leaf (neighbor(-ig,-jg,-kg)) &&\
  !(neighbor(-ig,-jg,-kg).pid < 0) &&\
  is_local(neighbor(-ig,-jg,-kg))) {\
       point.i -= ig; x -= ig*Delta/2.;\
\
       point.j -= jg; y -= jg*Delta/2.;\
\
\
       point.k -= kg; z -= kg*Delta/2.;\
\

#line 588

#define end_foreach_boundary()\
       point.i += ig; x += ig*Delta/2.;\
\
       point.j += jg; y += jg*Delta/2.;\
\
\
       point.k += kg; z += kg*Delta/2.;\
\
            }\
   }\
   ig = jg = kg = 0;\
 }\
    } end_foreach_boundary_level(); }\

#line 602


#define foreach_halo(_name,_l) {\
  if (_l <= depth()) {\
    { if (((Tree *)grid)->dirty) update_cache_f(); };\
    CacheLevel _cache = ((Tree *)grid)->_name[_l];\
    foreach_cache_level (_cache, _l)\

#line 609

#define end_foreach_halo() end_foreach_cache_level(); }}

#line 1 "grid/neighbors.h"
#line 1 "/home/miro3/Documents/Programming/basilisk/src/grid/neighbors.h"
#line 35 "/home/miro3/Documents/Programming/basilisk/src/grid/neighbors.h"
#define foreach_neighbor(_s) {\
  int _nn = _s + 0 ? _s + 0 : 2;\
  int _i = point.i, _j = point.j, _k = point.k;\
  for (int _l = - _nn; _l <= _nn; _l++) {\
    point.i = _i + _l;\
    for (int _m = - _nn; _m <= _nn; _m++) {\
      point.j = _j + _m;\
      for (int _n = - _nn; _n <= _nn; _n++) {\
 point.k = _k + _n;\
 POINT_VARIABLES;\

#line 45

#define end_foreach_neighbor()\
      }\
    }\
  }\
  point.i = _i; point.j = _j; point.k = _k;\
}\

#line 52

#define foreach_neighbor_break() _l = _m = _n = _nn + 1
#line 613 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"

static inline bool has_local_children (Point point)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 615 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"

   { foreach_child()
    if (is_local(cell))
      return true; end_foreach_child(); }
  return false;
}

static inline void cache_append_face (Point point, unsigned short flags)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 623 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"

  Tree * q = ((Tree *)grid);
  cache_append (&q->faces, point, flags);
#line 637 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
  {
#line 637

    if (flags & face_x)
      for (int i = 0; i <= 1; i++)
 for (int j = 0; j <= 1; j++)
   if (!(neighbor(0,i,j).flags & vertex)) {
     cache_append (&q->vertices, neighborp(0,i,j), 0);
     neighbor(0,i,j).flags |= vertex;
   }
#line 637

    if (flags & face_y)
      for (int i = 0; i <= 1; i++)
 for (int j = 0; j <= 1; j++)
   if (!(neighbor(j,0,i).flags & vertex)) {
     cache_append (&q->vertices, neighborp(j,0,i), 0);
     neighbor(j,0,i).flags |= vertex;
   }
#line 637

    if (flags & face_z)
      for (int i = 0; i <= 1; i++)
 for (int j = 0; j <= 1; j++)
   if (!(neighbor(i,j,0).flags & vertex)) {
     cache_append (&q->vertices, neighborp(i,j,0), 0);
     neighbor(i,j,0).flags |= vertex;
   }}

}



static void update_cache_f (void)
{
  Tree * q = ((Tree *)grid);

   { foreach_cache(q->vertices){

#line 654 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"

    if (level <= depth() && allocated(0,0,0))
      cell.flags &= ~vertex; } end_foreach_cache(); }


  q->leaves.n = q->faces.n = q->vertices.n = 0;
  for (int l = 0; l <= depth(); l++)
    q->active[l].n = q->prolongation[l].n =
      q->boundary[l].n = q->restriction[l].n = 0;

  const unsigned short fboundary = 1 << user;
   { foreach_cell(){

#line 665 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
 {



    if (is_local(cell) && is_active(cell)) {


      cache_level_append (&q->active[level], point);
    }
#line 689 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
    if (!(cell.pid < 0)) {

       { foreach_neighbor (BGHOSTS)
 if (allocated(0,0,0) && (cell.pid < 0) && !(cell.flags & fboundary)) {
   cache_level_append (&q->boundary[level], point);
   cell.flags |= fboundary;
 } end_foreach_neighbor(); }
    }

    else if (level > 0 && is_local(aparent(0,0,0)))
      cache_level_append (&q->restriction[level], point);

    if (is_leaf (cell)) {
      if (is_local(cell)) {
 cache_append (&q->leaves, point, 0);

 unsigned short flags = 0;
 {
#line 706

   if ((neighbor(-1,0,0).pid < 0) || (!is_leaf(neighbor(-1,0,0)) && !neighbor(-1,0,0).neighbors && neighbor(-1,0,0).pid >= 0) ||
       is_leaf(neighbor(-1,0,0)))
     flags |= face_x;
#line 706

   if ((neighbor(0,-1,0).pid < 0) || (!is_leaf(neighbor(0,-1,0)) && !neighbor(0,-1,0).neighbors && neighbor(0,-1,0).pid >= 0) ||
       is_leaf(neighbor(0,-1,0)))
     flags |= face_y;
#line 706

   if ((neighbor(0,0,-1).pid < 0) || (!is_leaf(neighbor(0,0,-1)) && !neighbor(0,0,-1).neighbors && neighbor(0,0,-1).pid >= 0) ||
       is_leaf(neighbor(0,0,-1)))
     flags |= face_z;}
 if (flags)
   cache_append (&q->faces, point, flags);
 {
#line 712

   if ((neighbor(1,0,0).pid < 0) || (!is_leaf(neighbor(1,0,0)) && !neighbor(1,0,0).neighbors && neighbor(1,0,0).pid >= 0) ||
       (!is_local(neighbor(1,0,0)) && is_leaf(neighbor(1,0,0))))
     cache_append (&q->faces, neighborp(1,0,0), face_x);
#line 712

   if ((neighbor(0,1,0).pid < 0) || (!is_leaf(neighbor(0,1,0)) && !neighbor(0,1,0).neighbors && neighbor(0,1,0).pid >= 0) ||
       (!is_local(neighbor(0,1,0)) && is_leaf(neighbor(0,1,0))))
     cache_append (&q->faces, neighborp(0,1,0), face_y);
#line 712

   if ((neighbor(0,0,1).pid < 0) || (!is_leaf(neighbor(0,0,1)) && !neighbor(0,0,1).neighbors && neighbor(0,0,1).pid >= 0) ||
       (!is_local(neighbor(0,0,1)) && is_leaf(neighbor(0,0,1))))
     cache_append (&q->faces, neighborp(0,0,1), face_z);}

 for (int i = 0; i <= 1; i++)

   for (int j = 0; j <= 1; j++)


     for (int k = 0; k <= 1; k++)

       if (!(neighbor(i,j,k).flags & vertex)) {
  cache_append (&q->vertices, neighborp(i,j,k), 0);
  neighbor(i,j,k).flags |= vertex;
       }

        if (cell.neighbors > 0)
   cache_level_append (&q->prolongation[level], point);
      }
      else if (!(cell.pid < 0) || is_local(aparent(0,0,0))) {

 unsigned short flags = 0;
 {
#line 735

   if (allocated(-1,0,0) &&
       is_local(neighbor(-1,0,0)) && (!is_leaf(neighbor(-1,0,0)) && !neighbor(-1,0,0).neighbors && neighbor(-1,0,0).pid >= 0))
     flags |= face_x;
#line 735

   if (allocated(0,-1,0) &&
       is_local(neighbor(0,-1,0)) && (!is_leaf(neighbor(0,-1,0)) && !neighbor(0,-1,0).neighbors && neighbor(0,-1,0).pid >= 0))
     flags |= face_y;
#line 735

   if (allocated(0,0,-1) &&
       is_local(neighbor(0,0,-1)) && (!is_leaf(neighbor(0,0,-1)) && !neighbor(0,0,-1).neighbors && neighbor(0,0,-1).pid >= 0))
     flags |= face_z;}
 if (flags)
   cache_append_face (point, flags);
 {
#line 741

   if (allocated(1,0,0) && is_local(neighbor(1,0,0)) &&
       (!is_leaf(neighbor(1,0,0)) && !neighbor(1,0,0).neighbors && neighbor(1,0,0).pid >= 0))
     cache_append_face (neighborp(1,0,0), face_x);
#line 741

   if (allocated(0,1,0) && is_local(neighbor(0,1,0)) &&
       (!is_leaf(neighbor(0,1,0)) && !neighbor(0,1,0).neighbors && neighbor(0,1,0).pid >= 0))
     cache_append_face (neighborp(0,1,0), face_y);
#line 741

   if (allocated(0,0,1) && is_local(neighbor(0,0,1)) &&
       (!is_leaf(neighbor(0,0,1)) && !neighbor(0,0,1).neighbors && neighbor(0,0,1).pid >= 0))
     cache_append_face (neighborp(0,0,1), face_z);}
      }

      continue;

    }
  } } end_foreach_cell(); }


  cache_shrink (&q->leaves);
  cache_shrink (&q->faces);
  cache_shrink (&q->vertices);
  for (int l = 0; l <= depth(); l++) {
    cache_level_shrink (&q->active[l]);
    cache_level_shrink (&q->prolongation[l]);
    cache_level_shrink (&q->boundary[l]);
    cache_level_shrink (&q->restriction[l]);
}

  q->dirty = false;


  for (int l = depth(); l >= 0; l--)
     { foreach_boundary_level (l){

#line 767 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"

      cell.flags &= ~fboundary; } end_foreach_boundary_level(); }



  grid->n = q->leaves.n;

#if !1
  grid->tn = grid->n;
  grid->maxdepth = grid->depth;
#endif
}

#define foreach() { if (((Tree *)grid)->dirty) update_cache_f(); }; foreach_cache(((Tree *)grid)->leaves)
#define end_foreach() end_foreach_cache()

#define foreach_face_generic()\
  { if (((Tree *)grid)->dirty) update_cache_f(); };\
  foreach_cache(((Tree *)grid)->faces) 
#line 784

#define end_foreach_face_generic() end_foreach_cache()

#define is_face_x() (_flags & face_x)

#define is_face_y() (_flags & face_y)


#define is_face_z() (_flags & face_z)


#define foreach_vertex()\
  { if (((Tree *)grid)->dirty) update_cache_f(); };\
  foreach_cache(((Tree *)grid)->vertices) {\
    x -= Delta/2.;\
\
    y -= Delta/2.;\
\
\
    z -= Delta/2.;\
\

#line 806

#define end_foreach_vertex() } end_foreach_cache()
#line 818 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
#define foreach_level(l) {\
  if (l <= depth()) {\
    { if (((Tree *)grid)->dirty) update_cache_f(); };\
    CacheLevel _active = ((Tree *)grid)->active[l];\
    foreach_cache_level (_active,l)\

#line 823

#define end_foreach_level() end_foreach_cache_level(); }}

#define foreach_coarse_level(l) foreach_level(l) if (!is_leaf(cell)) {
#define end_foreach_coarse_level() } end_foreach_level()

#define foreach_level_or_leaf(l) {\
  for (int _l1 = l; _l1 >= 0; _l1--)\
    foreach_level(_l1)\
      if (_l1 == l || is_leaf (cell)) {\

#line 833

#define end_foreach_level_or_leaf() } end_foreach_level(); }

#if TRASH
# undef trash
# define trash(list) reset(list, undefined)
#endif

void reset (void * alist, double val)
{
  scalar * list = (scalar *) alist;
  Tree * q = ((Tree *)grid);

  for (int l = 0; l <= depth(); l++) {
    Layer * L = q->L[l];
    for (int i = 0; i < L->len; i++)
      if (L->m[i])





 for (int j = 0; j < L->len; j++)
   if (L->m[i][j])





          for (int k = 0; k < L->len; k++)
     if (L->m[i][j][k])
       if (list) for (scalar s = *list, *_i10 = list; ((scalar *)&s)->i >= 0; s = *++_i10)
           if (!is_constant(s))
    ((double *)(L->m[i][j][k] + sizeof(Cell)))[s.i] = val;


  }
}

#define cache_level_resize(name, a)\
{\
  for (int i = 0; i <= depth() - a; i++)\
    pfree (q->name[i].p,__func__,__FILE__,__LINE__);\
  pfree (q->name,__func__,__FILE__,__LINE__);\
  q->name = ((CacheLevel *) pcalloc (depth() + 1, sizeof(CacheLevel),__func__,__FILE__,__LINE__));\
}\

#line 879


static void update_depth (int inc)
{
  Tree * q = ((Tree *)grid);
  grid->depth += inc;
  q->L = &(q->L[-1]);
  q->L = (Layer * *) prealloc (q->L, (grid->depth + 2)*sizeof(Layer *),__func__,__FILE__,__LINE__);
  q->L = &(q->L[1]);
  if (inc > 0)
    q->L[grid->depth] = new_layer (grid->depth);
  cache_level_resize (active, inc);
  cache_level_resize (prolongation, inc);
  cache_level_resize (boundary, inc);
  cache_level_resize (restriction, inc);
}

static void alloc_children (Point point)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 897 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"

  if (point.level == grid->depth)
    update_depth (+1);
  else if (allocated_child(0,0,0))
    return;
#line 929 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
  Layer * L = ((Tree *)grid)->L[point.level + 1];
  size_t len = sizeof(Cell) + datasize;
  char * b = (char *) mempool_alloc0 (L->pool);
  for (int k = 0; k < 2; k++)
    for (int l = 0; l < 2; l++) {
      layer_add_row (L, 2*point.i - 2 + k, 2*point.j - 2 + l);
      for (int n = 0; n < 2; n++) {
 assert (!CHILD(k,l,n));
 CHILD(k,l,n) = b;
 b += len;
      }
    }


  int pid = cell.pid;
   { foreach_child() {
    cell.pid = pid;
#if TRASH
    if (all) for (scalar s = *all, *_i11 = all; ((scalar *)&s)->i >= 0; s = *++_i11)
      val(s,0,0,0) = undefined;
#endif
  } end_foreach_child(); }
}
#line 985 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
static void free_children (Point point)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 986 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"


  Layer * L = ((Tree *)grid)->L[point.level + 1];
  assert (CHILD(0,0,0));
  mempool_free (L->pool, CHILD(0,0,0));
  for (int k = 0; k < 2; k++)
    for (int l = 0; l < 2; l++) {
      for (int n = 0; n < 2; n++)
 CHILD(k,l,n) = NULL;
      if (layer_remove_row (L, 2*point.i - 2 + k, 2*point.j - 2 + l)) {
 assert (point.level + 1 == grid->depth);
 update_depth (-1);
      }
    }
}


void increment_neighbors (Point point)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 1004 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"

  ((Tree *)grid)->dirty = true;
   { foreach_neighbor (2/2)
    if (cell.neighbors++ == 0)
      alloc_children (point); end_foreach_neighbor(); }
}

void decrement_neighbors (Point point)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 1012 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"

  ((Tree *)grid)->dirty = true;
   { foreach_neighbor (2/2)
    if (allocated(0,0,0)) {
      cell.neighbors--;
      if (cell.neighbors == 0)
 free_children (point);
    } end_foreach_neighbor(); }
  if (cell.neighbors) {
    int pid = cell.pid;
     { foreach_child() {
      cell.flags = 0;
      cell.pid = pid;
    } end_foreach_child(); }
  }
}

void realloc_scalar (void)
{

  Tree * q = ((Tree *)grid);
  size_t newlen = sizeof(Cell) + datasize;
  size_t oldlen = newlen - sizeof(double);

  size_t len = _size(0);
  for (int i = 0; i < len; i++)



    for (int j = 0; j < len; j++)



      for (int k = 0; k < len; k++)
 q->L[0]->m[i][j][k] = (char *) prealloc (q->L[0]->m[i][j][k], (newlen)*sizeof(char),__func__,__FILE__,__LINE__);



  for (int l = 1; l <= depth(); l++) {
    Layer * L = q->L[l];
    size_t len = L->len;
    Mempool * oldpool = L->pool;
    L->pool = mempool_new (poolsize (l, newlen), (1 << 3)*newlen);
    for (int i = 0; i < len; i += 2)
      if (L->m[i]) {
#line 1065 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
 for (int j = 0; j < len; j += 2)
   if (L->m[i][j]) {
#line 1076 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
     for (int k = 0; k < len; k += 2)
       if (L->m[i][j][k]) {
  char * new = (char *) mempool_alloc (L->pool);
  for (int l = 0; l < 2; l++)
    for (int m = 0; m < 2; m++)
      for (int n = 0; n < 2; n++) {
        memcpy (new, L->m[i+l][j+m][k+n], oldlen);
        L->m[i+l][j+m][k+n] = new;
        new += newlen;
      }
       }

   }

      }
    mempool_destroy (oldpool);
  }
}



#define VN v.x
#define VT v.y
#define VR v.z




#if 1
# define disable_fpe_for_mpi() disable_fpe (FE_DIVBYZERO|FE_INVALID)
# define enable_fpe_for_mpi() enable_fpe (FE_DIVBYZERO|FE_INVALID)
#else
# define disable_fpe_for_mpi()
# define enable_fpe_for_mpi()
#endif

static inline void no_restriction (Point point, scalar s);

static bool normal_neighbor (Point point, scalar * scalars, vector * vectors)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 1115 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"

  for (int k = 1; k <= BGHOSTS; k++)
    {
#line 1117

      for (int i = -k; i <= k; i += 2*k)
 if ((allocated(i,0,0) && !(neighbor(i,0,0).pid < 0))) {
   Point neighbor = neighborp(i,0,0);
   int id = (- cell.pid - 1);
   if (scalars) for (scalar s = *scalars, *_i12 = scalars; ((scalar *)&s)->i >= 0; s = *++_i12)
     val(s,0,0,0) = _attribute[s.i].boundary[id](neighbor, point, s);
   if (vectors) for (vector v = *vectors, *_i13 = vectors; ((scalar *)&v)->i >= 0; v = *++_i13) {
     scalar vn = VN;
     val(v.x,0,0,0) = _attribute[vn.i].boundary[id](neighbor, point, v.x);

     scalar vt = VT;
     val(v.y,0,0,0) = _attribute[vt.i].boundary[id](neighbor, point, v.y);


     scalar vr = VR;
     val(v.z,0,0,0) = _attribute[vr.i].boundary[id](neighbor, point, v.z);

   }
   return true;
 }
#line 1117

      for (int i = -k; i <= k; i += 2*k)
 if ((allocated(0,i,0) && !(neighbor(0,i,0).pid < 0))) {
   Point neighbor = neighborp(0,i,0);
   int id = (- cell.pid - 1);
   if (scalars) for (scalar s = *scalars, *_i12 = scalars; ((scalar *)&s)->i >= 0; s = *++_i12)
     val(s,0,0,0) = _attribute[s.i].boundary[id](neighbor, point, s);
   if (vectors) for (vector v = *vectors, *_i13 = vectors; ((scalar *)&v)->i >= 0; v = *++_i13) {
     scalar vn = VN;
     val(v.y,0,0,0) = _attribute[vn.i].boundary[id](neighbor, point, v.y);

     scalar vt = VT;
     val(v.z,0,0,0) = _attribute[vt.i].boundary[id](neighbor, point, v.z);


     scalar vr = VR;
     val(v.x,0,0,0) = _attribute[vr.i].boundary[id](neighbor, point, v.x);

   }
   return true;
 }
#line 1117

      for (int i = -k; i <= k; i += 2*k)
 if ((allocated(0,0,i) && !(neighbor(0,0,i).pid < 0))) {
   Point neighbor = neighborp(0,0,i);
   int id = (- cell.pid - 1);
   if (scalars) for (scalar s = *scalars, *_i12 = scalars; ((scalar *)&s)->i >= 0; s = *++_i12)
     val(s,0,0,0) = _attribute[s.i].boundary[id](neighbor, point, s);
   if (vectors) for (vector v = *vectors, *_i13 = vectors; ((scalar *)&v)->i >= 0; v = *++_i13) {
     scalar vn = VN;
     val(v.z,0,0,0) = _attribute[vn.i].boundary[id](neighbor, point, v.z);

     scalar vt = VT;
     val(v.x,0,0,0) = _attribute[vt.i].boundary[id](neighbor, point, v.x);


     scalar vr = VR;
     val(v.y,0,0,0) = _attribute[vr.i].boundary[id](neighbor, point, v.y);

   }
   return true;
 }}
  return false;
}

static bool diagonal_neighbor_2D (Point point,
      scalar * scalars, vector * vectors)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 1143 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"


  for (int k = 1; k <= BGHOSTS; k++)

    {
#line 1147


      for (int i = -k; i <= k; i += 2*k)
 for (int j = -k; j <= k; j += 2*k)
   if (allocated(i,j,0) && (allocated(i,j,0) && !(neighbor(i,j,0).pid < 0)) &&
       allocated(i,0,0) && (neighbor(i,0,0).pid < 0) &&
       allocated(0,j,0) && (neighbor(0,j,0).pid < 0)) {
     Point n = neighborp(i,j,0),
       n1 = neighborp(i,0,0), n2 = neighborp(0,j,0);
     int id1 = (- neighbor(i,0,0).pid - 1), id2 = (- neighbor(0,j,0).pid - 1);
     if (scalars) for (scalar s = *scalars, *_i14 = scalars; ((scalar *)&s)->i >= 0; s = *++_i14)
       val(s,0,0,0) = (_attribute[s.i].boundary[id1](n,n1,s) + _attribute[s.i].boundary[id2](n,n2,s) -
       val(s,i,j,0));
     if (vectors) for (vector v = *vectors, *_i15 = vectors; ((scalar *)&v)->i >= 0; v = *++_i15) {
       scalar vt = VT, vn = VN;
       val(v.x,0,0,0) = (_attribute[vt.i].boundary[id1](n,n1,v.x) +
         _attribute[vn.i].boundary[id2](n,n2,v.x) -
         val(v.x,i,j,0));
       val(v.y,0,0,0) = (_attribute[vn.i].boundary[id1](n,n1,v.y) +
         _attribute[vt.i].boundary[id2](n,n2,v.y) -
         val(v.y,i,j,0));

       scalar vr = VR;
       val(v.z,0,0,0) = (_attribute[vr.i].boundary[id1](n,n1,v.z) +
         _attribute[vr.i].boundary[id2](n,n2,v.z) -
         val(v.z,i,j,0));

     }
     return true;
   }
#line 1147


      for (int i = -k; i <= k; i += 2*k)
 for (int j = -k; j <= k; j += 2*k)
   if (allocated(0,i,j) && (allocated(0,i,j) && !(neighbor(0,i,j).pid < 0)) &&
       allocated(0,i,0) && (neighbor(0,i,0).pid < 0) &&
       allocated(0,0,j) && (neighbor(0,0,j).pid < 0)) {
     Point n = neighborp(0,i,j),
       n1 = neighborp(0,i,0), n2 = neighborp(0,0,j);
     int id1 = (- neighbor(0,i,0).pid - 1), id2 = (- neighbor(0,0,j).pid - 1);
     if (scalars) for (scalar s = *scalars, *_i14 = scalars; ((scalar *)&s)->i >= 0; s = *++_i14)
       val(s,0,0,0) = (_attribute[s.i].boundary[id1](n,n1,s) + _attribute[s.i].boundary[id2](n,n2,s) -
       val(s,0,i,j));
     if (vectors) for (vector v = *vectors, *_i15 = vectors; ((scalar *)&v)->i >= 0; v = *++_i15) {
       scalar vt = VT, vn = VN;
       val(v.y,0,0,0) = (_attribute[vt.i].boundary[id1](n,n1,v.y) +
         _attribute[vn.i].boundary[id2](n,n2,v.y) -
         val(v.y,0,i,j));
       val(v.z,0,0,0) = (_attribute[vn.i].boundary[id1](n,n1,v.z) +
         _attribute[vt.i].boundary[id2](n,n2,v.z) -
         val(v.z,0,i,j));

       scalar vr = VR;
       val(v.x,0,0,0) = (_attribute[vr.i].boundary[id1](n,n1,v.x) +
         _attribute[vr.i].boundary[id2](n,n2,v.x) -
         val(v.x,0,i,j));

     }
     return true;
   }
#line 1147


      for (int i = -k; i <= k; i += 2*k)
 for (int j = -k; j <= k; j += 2*k)
   if (allocated(j,0,i) && (allocated(j,0,i) && !(neighbor(j,0,i).pid < 0)) &&
       allocated(0,0,i) && (neighbor(0,0,i).pid < 0) &&
       allocated(j,0,0) && (neighbor(j,0,0).pid < 0)) {
     Point n = neighborp(j,0,i),
       n1 = neighborp(0,0,i), n2 = neighborp(j,0,0);
     int id1 = (- neighbor(0,0,i).pid - 1), id2 = (- neighbor(j,0,0).pid - 1);
     if (scalars) for (scalar s = *scalars, *_i14 = scalars; ((scalar *)&s)->i >= 0; s = *++_i14)
       val(s,0,0,0) = (_attribute[s.i].boundary[id1](n,n1,s) + _attribute[s.i].boundary[id2](n,n2,s) -
       val(s,j,0,i));
     if (vectors) for (vector v = *vectors, *_i15 = vectors; ((scalar *)&v)->i >= 0; v = *++_i15) {
       scalar vt = VT, vn = VN;
       val(v.z,0,0,0) = (_attribute[vt.i].boundary[id1](n,n1,v.z) +
         _attribute[vn.i].boundary[id2](n,n2,v.z) -
         val(v.z,j,0,i));
       val(v.x,0,0,0) = (_attribute[vn.i].boundary[id1](n,n1,v.x) +
         _attribute[vt.i].boundary[id2](n,n2,v.x) -
         val(v.x,j,0,i));

       scalar vr = VR;
       val(v.y,0,0,0) = (_attribute[vr.i].boundary[id1](n,n1,v.y) +
         _attribute[vr.i].boundary[id2](n,n2,v.y) -
         val(v.y,j,0,i));

     }
     return true;
   }}

  return false;
}

static bool diagonal_neighbor_3D (Point point,
      scalar * scalars, vector * vectors)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 1183 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"


  for (int n = 1; n <= BGHOSTS; n++)
    for (int i = -n; i <= n; i += 2*n)
      for (int j = -n; j <= n; j += 2*n)
 for (int k = -n; k <= n; k += 2*n)
   if ((allocated(i,j,k) && !(neighbor(i,j,k).pid < 0)) &&
       (neighbor(i,j,0).pid < 0) &&
       (neighbor(i,0,k).pid < 0) &&
       (neighbor(0,j,k).pid < 0)) {
     Point
       n0 = neighborp(i,j,k),
       n1 = neighborp(i,j,0),
       n2 = neighborp(i,0,k),
       n3 = neighborp(0,j,k);
     int
       id1 = (- neighbor(i,j,0).pid - 1),
       id2 = (- neighbor(i,0,k).pid - 1),
       id3 = (- neighbor(0,j,k).pid - 1);
     if (scalars) for (scalar s = *scalars, *_i16 = scalars; ((scalar *)&s)->i >= 0; s = *++_i16)
       val(s,0,0,0) = (_attribute[s.i].boundary[id1](n0,n1,s) +
       _attribute[s.i].boundary[id2](n0,n2,s) +
       _attribute[s.i].boundary[id3](n0,n3,s) -
       2.*val(s,i,j,k));
     if (vectors) for (vector v = *vectors, *_i17 = vectors; ((scalar *)&v)->i >= 0; v = *++_i17) {
       scalar vt = VT, vn = VN, vr = VR;
       val(v.x,0,0,0) = (_attribute[vt.i].boundary[id1](n0,n1,v.x) +
         _attribute[vt.i].boundary[id2](n0,n2,v.x) +
         _attribute[vn.i].boundary[id3](n0,n3,v.x) -
         2.*val(v.x,i,j,k));
       val(v.y,0,0,0) = (_attribute[vt.i].boundary[id1](n0,n1,v.y) +
         _attribute[vn.i].boundary[id2](n0,n2,v.y) +
         _attribute[vt.i].boundary[id3](n0,n3,v.y) -
         2.*val(v.y,i,j,k));
       val(v.z,0,0,0) = (_attribute[vn.i].boundary[id1](n0,n1,v.z) +
         _attribute[vr.i].boundary[id2](n0,n2,v.z) +
         _attribute[vr.i].boundary[id3](n0,n3,v.z) -
         2.*val(v.z,i,j,k));
     }
     return true;
   }

  return false;
}



#line 1229

static Point tangential_neighbor_x (Point point, bool * zn)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 1231 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"

  for (int k = 1; k <= BGHOSTS; k++)
    for (int j = -k; j <= k; j += 2*k) {
      if ((allocated(0,j,0) && !(neighbor(0,j,0).pid < 0)) || (allocated(-1,j,0) && !(neighbor(-1,j,0).pid < 0))) {
 *zn = false;
 return neighborp(0,j,0);
      }


      if ((allocated(0,0,j) && !(neighbor(0,0,j).pid < 0)) || (allocated(-1,0,j) && !(neighbor(-1,0,j).pid < 0))) {
 *zn = true;
 return neighborp(0,0,j);
      }

    }
  return (Point){.level = -1};
}
#line 1229

static Point tangential_neighbor_y (Point point, bool * zn)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 1231 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"

  for (int k = 1; k <= BGHOSTS; k++)
    for (int j = -k; j <= k; j += 2*k) {
      if ((allocated(0,0,j) && !(neighbor(0,0,j).pid < 0)) || (allocated(0,-1,j) && !(neighbor(0,-1,j).pid < 0))) {
 *zn = false;
 return neighborp(0,0,j);
      }


      if ((allocated(j,0,0) && !(neighbor(j,0,0).pid < 0)) || (allocated(j,-1,0) && !(neighbor(j,-1,0).pid < 0))) {
 *zn = true;
 return neighborp(j,0,0);
      }

    }
  return (Point){.level = -1};
}
#line 1229

static Point tangential_neighbor_z (Point point, bool * zn)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 1231 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"

  for (int k = 1; k <= BGHOSTS; k++)
    for (int j = -k; j <= k; j += 2*k) {
      if ((allocated(j,0,0) && !(neighbor(j,0,0).pid < 0)) || (allocated(j,0,-1) && !(neighbor(j,0,-1).pid < 0))) {
 *zn = false;
 return neighborp(j,0,0);
      }


      if ((allocated(0,j,0) && !(neighbor(0,j,0).pid < 0)) || (allocated(0,j,-1) && !(neighbor(0,j,-1).pid < 0))) {
 *zn = true;
 return neighborp(0,j,0);
      }

    }
  return (Point){.level = -1};
}


static inline bool is_boundary_point (Point point) { int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 1250 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"

  return (cell.pid < 0);
}

static void box_boundary_level (const Boundary * b, scalar * list, int l)
{
  disable_fpe_for_mpi();
  scalar * scalars = NULL;
  vector * vectors = NULL, * faces = NULL;
  if (list) for (scalar s = *list, *_i18 = list; ((scalar *)&s)->i >= 0; s = *++_i18)
    if (!is_constant(s) && _attribute[s.i].refine != no_restriction) {
      if (_attribute[s.i].v.x.i == s.i) {
 if (_attribute[s.i].face)
   faces = vectors_add (faces, _attribute[s.i].v);
 else
   vectors = vectors_add (vectors, _attribute[s.i].v);
      }
      else if (_attribute[s.i].v.x.i < 0)
 scalars = list_add (scalars, s);
    }

   { foreach_boundary_level (l){

#line 1271 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
 {
    if (!normal_neighbor (point, scalars, vectors) &&
 !diagonal_neighbor_2D (point, scalars, vectors) &&
 !diagonal_neighbor_3D (point, scalars, vectors)) {

      if (scalars) for (scalar s = *scalars, *_i19 = scalars; ((scalar *)&s)->i >= 0; s = *++_i19)
 val(s,0,0,0) = undefined;
      if (vectors) for (vector v = *vectors, *_i20 = vectors; ((scalar *)&v)->i >= 0; v = *++_i20)
 {
#line 1279

   val(v.x,0,0,0) = undefined;
#line 1279

   val(v.y,0,0,0) = undefined;
#line 1279

   val(v.z,0,0,0) = undefined;}
    }
    if (faces) {
      int id = (- cell.pid - 1);
      {
#line 1284

 for (int i = -1; i <= 1; i += 2) {

   if ((allocated(i,0,0) && !(neighbor(i,0,0).pid < 0))) {
     Point neighbor = neighborp(i,0,0);
     if (faces) for (vector v = *faces, *_i21 = faces; ((scalar *)&v)->i >= 0; v = *++_i21) {
       scalar vn = VN;
       if (_attribute[vn.i].boundary[id])
  val(v.x,(i + 1)/2,0,0) = _attribute[vn.i].boundary[id](neighbor, point, v.x);
     }
   }

   else if (i == -1) {

     bool zn;
     Point neighbor = tangential_neighbor_x (point, &zn);
     if (neighbor.level >= 0) {
       int id = is_boundary_point (neighbor) ?
  (- neighbor(-1,0,0).pid - 1) : (- cell.pid - 1);
       if (faces) for (vector v = *faces, *_i22 = faces; ((scalar *)&v)->i >= 0; v = *++_i22) {



  scalar vt = zn ? VT : VR;

  val(v.x,0,0,0) = _attribute[vt.i].boundary[id](neighbor, point, v.x);
       }
     }
     else

       if (faces) for (vector v = *faces, *_i23 = faces; ((scalar *)&v)->i >= 0; v = *++_i23)
  val(v.x,0,0,0) = 0.;
   }

 }
#line 1284

 for (int i = -1; i <= 1; i += 2) {

   if ((allocated(0,i,0) && !(neighbor(0,i,0).pid < 0))) {
     Point neighbor = neighborp(0,i,0);
     if (faces) for (vector v = *faces, *_i21 = faces; ((scalar *)&v)->i >= 0; v = *++_i21) {
       scalar vn = VN;
       if (_attribute[vn.i].boundary[id])
  val(v.y,0,(i + 1)/2,0) = _attribute[vn.i].boundary[id](neighbor, point, v.y);
     }
   }

   else if (i == -1) {

     bool zn;
     Point neighbor = tangential_neighbor_y (point, &zn);
     if (neighbor.level >= 0) {
       int id = is_boundary_point (neighbor) ?
  (- neighbor(0,-1,0).pid - 1) : (- cell.pid - 1);
       if (faces) for (vector v = *faces, *_i22 = faces; ((scalar *)&v)->i >= 0; v = *++_i22) {



  scalar vt = zn ? VT : VR;

  val(v.y,0,0,0) = _attribute[vt.i].boundary[id](neighbor, point, v.y);
       }
     }
     else

       if (faces) for (vector v = *faces, *_i23 = faces; ((scalar *)&v)->i >= 0; v = *++_i23)
  val(v.y,0,0,0) = 0.;
   }

 }
#line 1284

 for (int i = -1; i <= 1; i += 2) {

   if ((allocated(0,0,i) && !(neighbor(0,0,i).pid < 0))) {
     Point neighbor = neighborp(0,0,i);
     if (faces) for (vector v = *faces, *_i21 = faces; ((scalar *)&v)->i >= 0; v = *++_i21) {
       scalar vn = VN;
       if (_attribute[vn.i].boundary[id])
  val(v.z,0,0,(i + 1)/2) = _attribute[vn.i].boundary[id](neighbor, point, v.z);
     }
   }

   else if (i == -1) {

     bool zn;
     Point neighbor = tangential_neighbor_z (point, &zn);
     if (neighbor.level >= 0) {
       int id = is_boundary_point (neighbor) ?
  (- neighbor(0,0,-1).pid - 1) : (- cell.pid - 1);
       if (faces) for (vector v = *faces, *_i22 = faces; ((scalar *)&v)->i >= 0; v = *++_i22) {



  scalar vt = zn ? VT : VR;

  val(v.z,0,0,0) = _attribute[vt.i].boundary[id](neighbor, point, v.z);
       }
     }
     else

       if (faces) for (vector v = *faces, *_i23 = faces; ((scalar *)&v)->i >= 0; v = *++_i23)
  val(v.z,0,0,0) = 0.;
   }

 }}
    }
  } } end_foreach_boundary_level(); }

  pfree (scalars,__func__,__FILE__,__LINE__);
  pfree (vectors,__func__,__FILE__,__LINE__);
  pfree (faces,__func__,__FILE__,__LINE__);
  enable_fpe_for_mpi();
}



#undef VN
#undef VT
#define VN _attribute[s.i].v.x
#define VT _attribute[s.i].v.y

static double masked_average (Point point, scalar s)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 1336 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"

  double sum = 0., n = 0.;
   { foreach_child()
    if (!(cell.pid < 0) && val(s,0,0,0) != nodata)
      sum += val(s,0,0,0), n++; end_foreach_child(); }
  return n ? sum/n : nodata;
}


#line 1344

static double masked_average_x (Point point, scalar s)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 1346 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"

  double sum = 0., n = 0.;
   { foreach_child()
    if (child.x < 0 && (!(cell.pid < 0) || !(neighbor(1,0,0).pid < 0)) &&
 val(s,1,0,0) != nodata)
      sum += val(s,1,0,0), n++; end_foreach_child(); }
  return n ? sum/n : nodata;
}
#line 1344

static double masked_average_y (Point point, scalar s)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 1346 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"

  double sum = 0., n = 0.;
   { foreach_child()
    if (child.y < 0 && (!(cell.pid < 0) || !(neighbor(0,1,0).pid < 0)) &&
 val(s,0,1,0) != nodata)
      sum += val(s,0,1,0), n++; end_foreach_child(); }
  return n ? sum/n : nodata;
}
#line 1344

static double masked_average_z (Point point, scalar s)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 1346 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"

  double sum = 0., n = 0.;
   { foreach_child()
    if (child.z < 0 && (!(cell.pid < 0) || !(neighbor(0,0,1).pid < 0)) &&
 val(s,0,0,1) != nodata)
      sum += val(s,0,0,1), n++; end_foreach_child(); }
  return n ? sum/n : nodata;
}

static void masked_boundary_restriction (const Boundary * b,
      scalar * list, int l)
{
  scalar * scalars = NULL;
  vector * faces = NULL;
  if (list) for (scalar s = *list, *_i24 = list; ((scalar *)&s)->i >= 0; s = *++_i24)
    if (!is_constant(s) && _attribute[s.i].refine != no_restriction) {
      if (_attribute[s.i].v.x.i == s.i && _attribute[s.i].face)
 faces = vectors_add (faces, _attribute[s.i].v);
      else
 scalars = list_add (scalars, s);
    }

   { foreach_halo (restriction, l){

#line 1368 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
 {
    if (scalars) for (scalar s = *scalars, *_i25 = scalars; ((scalar *)&s)->i >= 0; s = *++_i25)
      val(s,0,0,0) = masked_average (parent, s);
    if (faces) for (vector v = *faces, *_i26 = faces; ((scalar *)&v)->i >= 0; v = *++_i26)
      {
#line 1372
 {
 double average = masked_average_x (parent, v.x);
 if ((neighbor(-1,0,0).pid < 0))
   val(v.x,0,0,0) = average;
 if ((neighbor(1,0,0).pid < 0))
   val(v.x,1,0,0) = average;
      }
#line 1372
 {
 double average = masked_average_y (parent, v.y);
 if ((neighbor(0,-1,0).pid < 0))
   val(v.y,0,0,0) = average;
 if ((neighbor(0,1,0).pid < 0))
   val(v.y,0,1,0) = average;
      }
#line 1372
 {
 double average = masked_average_z (parent, v.z);
 if ((neighbor(0,0,-1).pid < 0))
   val(v.z,0,0,0) = average;
 if ((neighbor(0,0,1).pid < 0))
   val(v.z,0,0,1) = average;
      }}
  } } end_foreach_halo(); }

  pfree (scalars,__func__,__FILE__,__LINE__);
  pfree (faces,__func__,__FILE__,__LINE__);
}
#line 1410 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
static double periodic_bc (Point, Point, scalar);


#line 1412

static void periodic_boundary_level_x (const Boundary * b, scalar * list, int l)
{
  scalar * list1 = NULL;
  if (list) for (scalar s = *list, *_i27 = list; ((scalar *)&s)->i >= 0; s = *++_i27)
    if (!is_constant(s)) {
      if (_attribute[s.i].face) {

 scalar vt = VT;
 if (_attribute[vt.i].boundary[right] == periodic_bc)
   list1 = list_add (list1, s);

      }
      else if (_attribute[s.i].boundary[right] == periodic_bc) {
 if (_attribute[s.i].v.x.i >= 0)
   {
#line 1427

     list1 = list_add (list1, _attribute[s.i].v.x);
#line 1427

     list1 = list_add (list1, _attribute[s.i].v.y);
#line 1427

     list1 = list_add (list1, _attribute[s.i].v.z);}
 else
   list1 = list_add (list1, s);
      }
    }
  if (!list1)
    return;

  if (l == 0) {
    Point point = {0};  int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 1437 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"

    int n = 1 << point.level;

    for (int j = 0; j < n + 2*2; j++)


      for (int k = 0; k < n + 2*2; k++)

 {
   for (int i = 0; i < 2; i++)
     if (list1) for (scalar s = *list1, *_i28 = list1; ((scalar *)&s)->i >= 0; s = *++_i28)
       val(s,i,j,k) = val(s,2,2,2);
   for (int i = n + 2; i < n + 2*2; i++)
     if (list1) for (scalar s = *list1, *_i29 = list1; ((scalar *)&s)->i >= 0; s = *++_i29)
       val(s,i,j,k) = val(s,2,2,2);
 }
  }
  else {
    OMP_PARALLEL() {



      Point point = {0,0,0};  int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 1459 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"


      point.level = l < 0 ? depth() : l;
      int n = 1 << point.level;

      int j;
      OMP(omp for schedule(static))
 for (j = 0; j < n + 2*2; j++)


   for (int k = 0; k < n + 2*2; k++)

     {
       for (int i = 0; i < 2; i++)
  if (allocated(i + n,j,k))
    if (list1) for (scalar s = *list1, *_i30 = list1; ((scalar *)&s)->i >= 0; s = *++_i30)
      val(s,i,j,k) = val(s,i + n,j,k);
       for (int i = n + 2; i < n + 2*2; i++)
  if (allocated(i - n,j,k))
    if (list1) for (scalar s = *list1, *_i31 = list1; ((scalar *)&s)->i >= 0; s = *++_i31)
      val(s,i,j,k) = val(s,i - n,j,k);
     }
    }
  }

  pfree (list1,__func__,__FILE__,__LINE__);
}
#line 1412

static void periodic_boundary_level_y (const Boundary * b, scalar * list, int l)
{
  scalar * list1 = NULL;
  if (list) for (scalar s = *list, *_i27 = list; ((scalar *)&s)->i >= 0; s = *++_i27)
    if (!is_constant(s)) {
      if (_attribute[s.i].face) {

 scalar vt = VT;
 if (_attribute[vt.i].boundary[top] == periodic_bc)
   list1 = list_add (list1, s);

      }
      else if (_attribute[s.i].boundary[top] == periodic_bc) {
 if (_attribute[s.i].v.y.i >= 0)
   {
#line 1427

     list1 = list_add (list1, _attribute[s.i].v.y);
#line 1427

     list1 = list_add (list1, _attribute[s.i].v.z);
#line 1427

     list1 = list_add (list1, _attribute[s.i].v.x);}
 else
   list1 = list_add (list1, s);
      }
    }
  if (!list1)
    return;

  if (l == 0) {
    Point point = {0};  int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 1437 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"

    int n = 1 << point.level;

    for (int j = 0; j < n + 2*2; j++)


      for (int k = 0; k < n + 2*2; k++)

 {
   for (int i = 0; i < 2; i++)
     if (list1) for (scalar s = *list1, *_i28 = list1; ((scalar *)&s)->i >= 0; s = *++_i28)
       val(s,k,i,j) = val(s,2,2,2);
   for (int i = n + 2; i < n + 2*2; i++)
     if (list1) for (scalar s = *list1, *_i29 = list1; ((scalar *)&s)->i >= 0; s = *++_i29)
       val(s,k,i,j) = val(s,2,2,2);
 }
  }
  else {
    OMP_PARALLEL() {



      Point point = {0,0,0};  int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 1459 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"


      point.level = l < 0 ? depth() : l;
      int n = 1 << point.level;

      int j;
      OMP(omp for schedule(static))
 for (j = 0; j < n + 2*2; j++)


   for (int k = 0; k < n + 2*2; k++)

     {
       for (int i = 0; i < 2; i++)
  if (allocated(k,i + n,j))
    if (list1) for (scalar s = *list1, *_i30 = list1; ((scalar *)&s)->i >= 0; s = *++_i30)
      val(s,k,i,j) = val(s,k,i + n,j);
       for (int i = n + 2; i < n + 2*2; i++)
  if (allocated(k,i - n,j))
    if (list1) for (scalar s = *list1, *_i31 = list1; ((scalar *)&s)->i >= 0; s = *++_i31)
      val(s,k,i,j) = val(s,k,i - n,j);
     }
    }
  }

  pfree (list1,__func__,__FILE__,__LINE__);
}
#line 1412

static void periodic_boundary_level_z (const Boundary * b, scalar * list, int l)
{
  scalar * list1 = NULL;
  if (list) for (scalar s = *list, *_i27 = list; ((scalar *)&s)->i >= 0; s = *++_i27)
    if (!is_constant(s)) {
      if (_attribute[s.i].face) {

 scalar vt = VT;
 if (_attribute[vt.i].boundary[front] == periodic_bc)
   list1 = list_add (list1, s);

      }
      else if (_attribute[s.i].boundary[front] == periodic_bc) {
 if (_attribute[s.i].v.z.i >= 0)
   {
#line 1427

     list1 = list_add (list1, _attribute[s.i].v.z);
#line 1427

     list1 = list_add (list1, _attribute[s.i].v.x);
#line 1427

     list1 = list_add (list1, _attribute[s.i].v.y);}
 else
   list1 = list_add (list1, s);
      }
    }
  if (!list1)
    return;

  if (l == 0) {
    Point point = {0};  int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 1437 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"

    int n = 1 << point.level;

    for (int j = 0; j < n + 2*2; j++)


      for (int k = 0; k < n + 2*2; k++)

 {
   for (int i = 0; i < 2; i++)
     if (list1) for (scalar s = *list1, *_i28 = list1; ((scalar *)&s)->i >= 0; s = *++_i28)
       val(s,j,k,i) = val(s,2,2,2);
   for (int i = n + 2; i < n + 2*2; i++)
     if (list1) for (scalar s = *list1, *_i29 = list1; ((scalar *)&s)->i >= 0; s = *++_i29)
       val(s,j,k,i) = val(s,2,2,2);
 }
  }
  else {
    OMP_PARALLEL() {



      Point point = {0,0,0};  int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 1459 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"


      point.level = l < 0 ? depth() : l;
      int n = 1 << point.level;

      int j;
      OMP(omp for schedule(static))
 for (j = 0; j < n + 2*2; j++)


   for (int k = 0; k < n + 2*2; k++)

     {
       for (int i = 0; i < 2; i++)
  if (allocated(j,k,i + n))
    if (list1) for (scalar s = *list1, *_i30 = list1; ((scalar *)&s)->i >= 0; s = *++_i30)
      val(s,j,k,i) = val(s,j,k,i + n);
       for (int i = n + 2; i < n + 2*2; i++)
  if (allocated(j,k,i - n))
    if (list1) for (scalar s = *list1, *_i31 = list1; ((scalar *)&s)->i >= 0; s = *++_i31)
      val(s,j,k,i) = val(s,j,k,i - n);
     }
    }
  }

  pfree (list1,__func__,__FILE__,__LINE__);
}

#undef VN
#undef VT

static void free_cache (CacheLevel * c)
{
  for (int l = 0; l <= depth(); l++)
    pfree (c[l].p,__func__,__FILE__,__LINE__);
  pfree (c,__func__,__FILE__,__LINE__);
}

void free_grid (void)
{
  if (!grid)
    return;
  free_boundaries();
  Tree * q = ((Tree *)grid);
  pfree (q->leaves.p,__func__,__FILE__,__LINE__);
  pfree (q->faces.p,__func__,__FILE__,__LINE__);
  pfree (q->vertices.p,__func__,__FILE__,__LINE__);
  pfree (q->refined.p,__func__,__FILE__,__LINE__);


  Layer * L = q->L[0];
#line 1527 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
  for (int i = 0; i < L->len; i++) {
    for (int j = 0; j < L->len; j++) {
      for (int k = 0; k < L->len; k++)
 pfree (L->m[i][j][k],__func__,__FILE__,__LINE__);
      pfree (L->m[i][j],__func__,__FILE__,__LINE__);
    }
    pfree (L->m[i],__func__,__FILE__,__LINE__);
  }

  for (int l = 1; l <= depth(); l++) {
    Layer * L = q->L[l];
    for (int i = 0; i < L->len; i++)
      if (L->m[i]) {
 for (int j = 0; j < L->len; j++)
   if (L->m[i][j])
     pfree (L->m[i][j],__func__,__FILE__,__LINE__);
 pfree (L->m[i],__func__,__FILE__,__LINE__);
      }
  }

  for (int l = 0; l <= depth(); l++)
    destroy_layer (q->L[l]);
  q->L = &(q->L[-1]);
  pfree (q->L,__func__,__FILE__,__LINE__);
  free_cache (q->active);
  free_cache (q->prolongation);
  free_cache (q->boundary);
  free_cache (q->restriction);
  pfree (q,__func__,__FILE__,__LINE__);
  grid = NULL;
}

static void refine_level (int depth);


void init_grid (int n)
{ trace ("init_grid", "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h", 1563);

  assert (sizeof(Cell) % 8 == 0);

  free_grid();
  int depth = 0;
  while (n > 1) {
    if (n % 2) {
      fprintf (qstderr(), "tree: N must be a power-of-two\n");
      exit (1);
    }
    n /= 2;
    depth++;
  }
  Tree * q = ((Tree *) pcalloc (1, sizeof(Tree),__func__,__FILE__,__LINE__));
  grid = (Grid *) q;
  grid->depth = 0;


  q->L = ((Layer * *) pmalloc ((2)*sizeof(Layer *),__func__,__FILE__,__LINE__));

  q->L[0] = NULL; q->L = &(q->L[1]);

  Layer * L = new_layer (0);
  q->L[0] = L;
#line 1619 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
  for (int i = 0; i < L->len; i++)
    for (int j = 0; j < L->len; j++) {
      layer_add_row (L, i, j);
      for (int k = 0; k < L->len; k++)
 L->m[i][j][k] = (char *) pcalloc (1, sizeof(Cell) + datasize,__func__,__FILE__,__LINE__);
    }
  CELL(L->m[2][2][2]).flags |= leaf;
  if (pid() == 0)
    CELL(L->m[2][2][2]).flags |= active;
  for (int k = -2; k <= 2; k++)
    for (int l = -2; l <= 2; l++)
      for (int n = -2; n <= 2; n++)
 CELL(L->m[2 +k][2 +l][2 +n]).pid =
   (k > 0 ? -1 - right :
    k < 0 ? -1 - left :
    l > 0 ? -1 - top :
    l < 0 ? -1 - bottom :
    n > 0 ? -1 - front :
    n < 0 ? -1 - back :
    0);

  q->active = ((CacheLevel *) pcalloc (1, sizeof(CacheLevel),__func__,__FILE__,__LINE__));
  q->prolongation = ((CacheLevel *) pcalloc (1, sizeof(CacheLevel),__func__,__FILE__,__LINE__));
  q->boundary = ((CacheLevel *) pcalloc (1, sizeof(CacheLevel),__func__,__FILE__,__LINE__));
  q->restriction = ((CacheLevel *) pcalloc (1, sizeof(CacheLevel),__func__,__FILE__,__LINE__));
  q->dirty = true;
  N = 1 << depth;
#if 1
  void mpi_boundary_new();
  mpi_boundary_new();
#endif
  { if (((Tree *)grid)->dirty) update_cache_f(); };

  Boundary * b = ((Boundary *) pcalloc (1, sizeof(Boundary),__func__,__FILE__,__LINE__));
  b->level = box_boundary_level;
  b->restriction = masked_boundary_restriction;
  add_boundary (b);

  {
#line 1657
 {
    Boundary * b = ((Boundary *) pcalloc (1, sizeof(Boundary),__func__,__FILE__,__LINE__));
    b->level = periodic_boundary_level_x;
    add_boundary (b);
  }
#line 1657
 {
    Boundary * b = ((Boundary *) pcalloc (1, sizeof(Boundary),__func__,__FILE__,__LINE__));
    b->level = periodic_boundary_level_y;
    add_boundary (b);
  }
#line 1657
 {
    Boundary * b = ((Boundary *) pcalloc (1, sizeof(Boundary),__func__,__FILE__,__LINE__));
    b->level = periodic_boundary_level_z;
    add_boundary (b);
  }}
  refine_level (depth);
  reset (all, 0.);
 end_trace("init_grid", "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h", 1664); }
#line 1700 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
struct _locate { double x, y, z; };

Point locate (struct _locate p)
{
  for (int l = depth(); l >= 0; l--) {
    Point point = { .level = l };
    int n = 1 << point.level;
    point.i = (p.x - X0)/L0*n + 2;

    point.j = (p.y - Y0)/L0*n + 2;


    point.k = (p.z - Z0)/L0*n + 2;

    if (point.i >= 0 && point.i < n + 2*2

 && point.j >= 0 && point.j < n + 2*2


 && point.k >= 0 && point.k < n + 2*2

 ) {
      if (allocated(0,0,0) && is_local(cell) && is_leaf(cell))
 return point;
    }
    else
      break;
  }
  Point point = { .level = -1 };
  return point;
}



bool tree_is_full()
{
  { if (((Tree *)grid)->dirty) update_cache_f(); };
  return (grid->tn == 1L << grid->maxdepth*3);
}

#line 1 "grid/tree-common.h"
#line 1 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"



#line 1 "grid/multigrid-common.h"
#line 1 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"


#line 1 "grid/cartesian-common.h"
#line 1 "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h"
#line 1 "grid/events.h"
#line 1 "/home/miro3/Documents/Programming/basilisk/src/grid/events.h"







static void event_error (Event * ev, const char * s)
{
  fprintf (qstderr(), "%s:%d: error: %s\n", ev->file, ev->line, s);
  exit (1);
}

static void init_event (Event * ev)
{
  if (ev->arrayi || ev->arrayt) {
    ev->i = ev->t = -1;
    if (ev->arrayi)
      ev->i = ev->arrayi[0];
    else
      ev->t = ev->arrayt[0];
    ev->a = 1;
    ev->expr[1] = NULL;
  }
  else {
    if (ev->nexpr > 0) {
      Expr init = NULL, cond = NULL, inc = NULL;
      for (int j = 0; j < ev->nexpr; j++) {
 int i = -123456; double t = i;
 (* ev->expr[j]) (&i, &t, ev);
 if (i == -123456 && t == -123456) {

   if (cond)
     event_error (ev, "events can only use a single condition");
   cond = ev->expr[j];
 }
 else {

   int i1 = i; double t1 = t;
   (* ev->expr[j]) (&i1, &t1, ev);
   if (i1 == i && t1 == t) {


     if (init)
       event_error (ev, "events can only use a single initialisation");
     init = ev->expr[j];
   }
   else {

     if (inc)
       event_error (ev, "events can only use a single increment");
     inc = ev->expr[j];
   }
 }
      }
      ev->expr[0] = init;
      ev->expr[1] = cond;
      ev->expr[2] = inc;
      ev->nexpr = 0;
    }
    ev->i = ev->t = -1;
    if (ev->expr[0]) {
      (* ev->expr[0]) (&ev->i, &ev->t, ev);
      if (ev->i == 1234567890 || ev->t == 1234567890) {
 ev->i = 1234567890; ev->t = -1;
      }
    }
    else if (ev->expr[2]) {
      (* ev->expr[2]) (&ev->i, &ev->t, ev);
      if (ev->i != -1)
 ev->i = 0;
      if (ev->t != -1)
 ev->t = 0;
    }
  }
}

enum { event_done, event_alive, event_stop };

static int event_finished (Event * ev)
{
  ev->t = ev->i = -1;
  return event_done;
}

void event_register (Event event) {
  assert (Events);
  assert (!event.last);
  int n = 0, parent = -1;
  for (Event * ev = Events; !ev->last; ev++) {
    if (!strcmp (event.name, ev->name)) {
      assert (parent < 0);
      parent = n;
    }
    n++;
  }
  if (parent < 0) {
    Events = (Event *) prealloc (Events, (n + 2)*sizeof(Event),__func__,__FILE__,__LINE__);
    Events[n] = event;
    Events[n].next = NULL;
    Events[n + 1].last = true;
    init_event (&Events[n]);
  }
  else {
    Event * ev = ((Event *) pcalloc (1, sizeof(Event),__func__,__FILE__,__LINE__));
    *ev = Events[parent];
    Events[parent] = event;
    Events[parent].next = ev;
    init_event (&Events[parent]);
  }
}

static int event_cond (Event * ev, int i, double t)
{
  if (!ev->expr[1])
    return true;
  return (* ev->expr[1]) (&i, &t, ev);
}
#line 131 "/home/miro3/Documents/Programming/basilisk/src/grid/events.h"
static int event_do (Event * ev, bool action)
{
  if ((iter > ev->i && t > ev->t) || !event_cond (ev, iter, t))
    return event_finished (ev);
  if (iter == ev->i || fabs (t - ev->t) <= 1e-9) {
    if (action) {
      bool finished = false;
      for (Event * e = ev; e; e = e->next) {



 if ((* e->action) (iter, t, e))
   finished = true;
      }
      if (finished) {
 event_finished (ev);
 return event_stop;
      }
    }
    if (ev->arrayi) {
      ev->i = ev->arrayi[ev->a++];
      if (ev->i < 0)
 return event_finished (ev);
    }
    if (ev->arrayt) {
      ev->t = ev->arrayt[ev->a++];
      if (ev->t < 0)
 return event_finished (ev);
    }
    else if (ev->expr[2]) {
      int i0 = ev->i;
      (* ev->expr[2]) (&ev->i, &ev->t, ev);
      if (i0 == -1 && ev->i != i0)
 ev->i += iter + 1;
      if (!event_cond (ev, iter + 1, ev->t))
 return event_finished (ev);
    }
    else if (ev->expr[0] && !ev->expr[1])
      return event_finished (ev);
  }
  return event_alive;
}

static void end_event_do (bool action)
{




  for (Event * ev = Events; !ev->last; ev++)
    if (ev->i == 1234567890 && action)
      for (Event * e = ev; e; e = e->next) {



 e->action (iter, t, e);
      }
}

int events (bool action)
{





  if (iter == 0)
    for (Event * ev = Events; !ev->last; ev++)
      init_event (ev);

  int cond = 0, cond1 = 0;
  inext = 1234567890; tnext = HUGE;
  for (Event * ev = Events; !ev->last && !cond; ev++)
    if (ev->i != 1234567890 &&
 (ev->expr[1] || (ev->expr[0] && !ev->expr[1] && !ev->expr[2]) || ev->arrayi || ev->arrayt))
      cond = 1;
  for (Event * ev = Events; !ev->last; ev++) {
    int status = event_do (ev, action);
    if (status == event_stop) {
      end_event_do (action);
      return 0;
    }
    if (status == event_alive && ev->i != 1234567890 &&
 (ev->expr[1] || (ev->expr[0] && !ev->expr[1] && !ev->expr[2]) || ev->arrayi || ev->arrayt))
      cond1 = 1;
    if (ev->t > t && ev->t < tnext)
      tnext = ev->t;
    if (ev->i > iter && ev->i < inext)
      inext = ev->i;
  }
  if ((!cond || cond1) && (tnext != HUGE || inext != 1234567890)) {
    inext = iter + 1;
    return 1;
  }
  end_event_do (action);
  return 0;
}

void event (const char * name)
{
  for (Event * ev = Events; !ev->last; ev++)
    if (!strcmp (ev->name, name))
      for (Event * e = ev; e; e = e->next) {



 (* e->action) (0, 0, e);
      }
}

double dtnext (double dt)
{
  if (tnext != HUGE && tnext > t) {
    unsigned int n = (tnext - t)/dt;
    assert (n < INT_MAX);
    if (n == 0)
      dt = tnext - t;
    else {
      double dt1 = (tnext - t)/n;
      if (dt1 > dt + 1e-9)
 dt = (tnext - t)/(n + 1);
      else if (dt1 < dt)
 dt = dt1;
      tnext = t + dt;
    }
  }
  else
    tnext = t + dt;
  return dt;
}
#line 2 "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h"

void (* debug) (Point);

#define _val_constant(a,k,l,m) ((const double) _constant[a.i -_NVARMAX])

#undef VARIABLES
#define VARIABLES\
  double Delta = L0*(1./(1 << point.level));\
  \
    double Delta_x = Delta;\
    double Delta_y = Delta;\
    double Delta_z = Delta;\
\
  double x = (ig/2. + (point.i - 2) + 0.5)*Delta + X0; NOT_UNUSED(x);\
\
  double y = (jg/2. + (point.j - 2) + 0.5)*Delta + Y0;\
\
\
\
 NOT_UNUSED(y);\
\
  double z = (kg/2. + (point.k - 2) + 0.5)*Delta + Z0;\
\
\
\
  NOT_UNUSED(z);\
\
  NOT_UNUSED(Delta);\
  \
    NOT_UNUSED(Delta_x);\
    NOT_UNUSED(Delta_y);\
    NOT_UNUSED(Delta_z);\
\
  ;\

#line 32


#line 1 "grid/fpe.h"
#line 1 "/home/miro3/Documents/Programming/basilisk/src/grid/fpe.h"


#include <signal.h>
#include <unistd.h>

static void gdb()
{
  if (last_point.level >= 0) {
    debug (last_point);
    fputc ('\n', qstderr());
    fflush (qstderr());
  }
  char command[80];
  sprintf (command, "exec xterm -e 'gdb -p %d' & xterm -e 'gnuplot plot -'",
    getpid());
  system (command);
}

static void caught_abort (int sig)
{
  fprintf (qstderr(), "Caught signal %d (Aborted)\n", sig);
  gdb();
}

static void caught_fpe (int sig)
{
  fprintf (qstderr(), "Caught signal %d (Floating Point Exception)\n", sig);
  gdb();
  exit (1);
}

static void caught_segfault (int sig)
{
  fprintf (qstderr(), "Caught signal %d (Segmentation Fault)\n", sig);
  gdb();
  exit (2);
}

void catch_fpe (void)
{
  struct sigaction act;
  act.sa_handler = caught_fpe;
  sigemptyset (&act.sa_mask);
  act.sa_flags = 0;
  last_point.level = -1;
  sigaction (8, &act, NULL);
  act.sa_handler = caught_segfault;
  sigaction (11, &act, NULL);
  act.sa_handler = caught_abort;
  act.sa_flags = SA_RESETHAND;
  sigaction (6, &act, NULL);
}
#line 35 "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h"

#define end_foreach_face()

scalar new_scalar (const char * name)
{
  int nvar = datasize/sizeof(double);
  scalar s;
  for (s.i = 0; s.i < nvar; s.i++)
    if (!list_lookup (all, s)) {
      init_scalar (s, name);
      trash (((scalar []){s, {-1}}));
      all = list_append (all, s);
      return s;
    }


  assert (nvar < _NVARMAX);
  datasize += sizeof(double); nvar++;
  _attribute = (_Attributes *) prealloc (_attribute, (nvar)*sizeof(_Attributes),__func__,__FILE__,__LINE__);
  memset (&_attribute[nvar-1], 0, sizeof (_Attributes));
  s = (scalar){nvar - 1};
  realloc_scalar();
  init_scalar (s, name);
  trash (((scalar []){s, {-1}}));
  all = list_append (all, s);
  return s;
}

scalar new_vertex_scalar (const char * name)
{
  scalar s = new_scalar (name);
  {
#line 66

    _attribute[s.i].d.x = -1;
#line 66

    _attribute[s.i].d.y = -1;
#line 66

    _attribute[s.i].d.z = -1;}
  return s;
}

static vector alloc_vector (const char * name)
{
  vector v;
  char cname[strlen(name) + 3];
  struct { char * x, * y, * z; } ext = {"%s.x", "%s.y", "%s.z"};
  {
#line 76
 {
    sprintf (cname, ext.x, name);
    v.x = new_scalar (cname);
  }
#line 76
 {
    sprintf (cname, ext.y, name);
    v.y = new_scalar (cname);
  }
#line 76
 {
    sprintf (cname, ext.z, name);
    v.z = new_scalar (cname);
  }}
  return v;
}

vector new_vector (const char * name)
{
  vector v = alloc_vector (name);
  init_vector (v, NULL);
  return v;
}

vector new_face_vector (const char * name)
{
  vector v = alloc_vector (name);
  init_face_vector (v, NULL);
  return v;
}

tensor new_tensor (const char * name)
{
  char cname[strlen(name) + 3];
  struct { char * x, * y, * z; } ext = {"%s.x", "%s.y", "%s.z"};
  tensor t;
  {
#line 102
 {
    sprintf (cname, ext.x, name);
    t.x = new_vector (cname);
  }
#line 102
 {
    sprintf (cname, ext.y, name);
    t.y = new_vector (cname);
  }
#line 102
 {
    sprintf (cname, ext.z, name);
    t.z = new_vector (cname);
  }}
  init_tensor (t, NULL);
  return t;
}

tensor new_symmetric_tensor (const char * name)
{
  char cname[strlen(name) + 5];
  struct { char * x, * y, * z; } ext = {"%s.x.x", "%s.y.y", "%s.z.z"};
  tensor t;
  {
#line 115
 {
    sprintf (cname, ext.x, name);
    t.x.x = new_scalar(cname);
  }
#line 115
 {
    sprintf (cname, ext.y, name);
    t.y.y = new_scalar(cname);
  }
#line 115
 {
    sprintf (cname, ext.z, name);
    t.z.z = new_scalar(cname);
  }}

    sprintf (cname, "%s.x.y", name);
    t.x.y = new_scalar(cname);
    t.y.x = t.x.y;


    sprintf (cname, "%s.x.z", name);
    t.x.z = new_scalar(cname);
    t.z.x = t.x.z;
    sprintf (cname, "%s.y.z", name);
    t.y.z = new_scalar(cname);
    t.z.y = t.y.z;




  init_tensor (t, NULL);
  return t;
}

static int nconst = 0;

void init_const_scalar (scalar s, const char * name, double val)
{
  if (s.i - _NVARMAX >= nconst) {
    nconst = s.i - _NVARMAX + 1;
    _constant = (double *) prealloc (_constant, (nconst)*sizeof(double),__func__,__FILE__,__LINE__);
  }
  _constant[s.i - _NVARMAX] = val;
}

scalar new_const_scalar (const char * name, int i, double val)
{
  scalar s = (scalar){i + _NVARMAX};
  init_const_scalar (s, name, val);
  return s;
}

void init_const_vector (vector v, const char * name, double * val)
{
  {
#line 159

    init_const_scalar (v.x, name, *val++);
#line 159

    init_const_scalar (v.y, name, *val++);
#line 159

    init_const_scalar (v.z, name, *val++);}
}

vector new_const_vector (const char * name, int i, double * val)
{
  vector v;
  {
#line 166

    v.x.i = _NVARMAX + i++;
#line 166

    v.y.i = _NVARMAX + i++;
#line 166

    v.z.i = _NVARMAX + i++;}
  init_const_vector (v, name, val);
  return v;
}

void scalar_clone (scalar a, scalar b)
{
  char * name = _attribute[a.i].name;
  double (** boundary) (Point, Point, scalar) = _attribute[a.i].boundary;
  double (** boundary_homogeneous) (Point, Point, scalar) =
    _attribute[a.i].boundary_homogeneous;
  _attribute[a.i] = _attribute[b.i];
  _attribute[a.i].name = name;
  _attribute[a.i].boundary = boundary;
  _attribute[a.i].boundary_homogeneous = boundary_homogeneous;
  for (int i = 0; i < nboundary; i++) {
    _attribute[a.i].boundary[i] = _attribute[b.i].boundary[i];
    _attribute[a.i].boundary_homogeneous[i] = _attribute[b.i].boundary_homogeneous[i];
  }
}

scalar * list_clone (scalar * l)
{
  scalar * list = NULL;
  int nvar = datasize/sizeof(double), map[nvar];
  for (int i = 0; i < nvar; i++)
    map[i] = -1;
  if (l) for (scalar s = *l, *_i32 = l; ((scalar *)&s)->i >= 0; s = *++_i32) {
    scalar c = new_scalar("c");
    scalar_clone (c, s);
    map[s.i] = c.i;
    list = list_append (list, c);
  }
  if (list) for (scalar s = *list, *_i33 = list; ((scalar *)&s)->i >= 0; s = *++_i33)
    {
#line 201

      if (_attribute[s.i].v.x.i >= 0 && map[_attribute[s.i].v.x.i] >= 0)
 _attribute[s.i].v.x.i = map[_attribute[s.i].v.x.i];
#line 201

      if (_attribute[s.i].v.y.i >= 0 && map[_attribute[s.i].v.y.i] >= 0)
 _attribute[s.i].v.y.i = map[_attribute[s.i].v.y.i];
#line 201

      if (_attribute[s.i].v.z.i >= 0 && map[_attribute[s.i].v.z.i] >= 0)
 _attribute[s.i].v.z.i = map[_attribute[s.i].v.z.i];}
  return list;
}

void delete (scalar * list)
{
  if (all == NULL)
    return;

  if (list) for (scalar f = *list, *_i34 = list; ((scalar *)&f)->i >= 0; f = *++_i34) {
    if (_attribute[f.i].delete)
      _attribute[f.i].delete (f);
    pfree (_attribute[f.i].name,__func__,__FILE__,__LINE__); _attribute[f.i].name = NULL;
    pfree (_attribute[f.i].boundary,__func__,__FILE__,__LINE__); _attribute[f.i].boundary = NULL;
    pfree (_attribute[f.i].boundary_homogeneous,__func__,__FILE__,__LINE__); _attribute[f.i].boundary_homogeneous = NULL;
  }

  if (list == all) {
    all[0].i = -1;
    return;
  }

  trash (list);
  if (list) for (scalar f = *list, *_i35 = list; ((scalar *)&f)->i >= 0; f = *++_i35) {
    scalar * s = all;
    for (; s->i >= 0 && s->i != f.i; s++);
    if (s->i == f.i)
      for (; s->i >= 0; s++)
 s[0] = s[1];
  }
}

typedef void (* free_solver_func) (void);

static Array * free_solver_funcs = NULL;

void free_solver_func_add (free_solver_func func)
{
  if (!free_solver_funcs)
    free_solver_funcs = array_new();
  array_append (free_solver_funcs, &func, sizeof(free_solver_func));
}

void free_solver()
{
  if (free_solver_funcs) {
    free_solver_func * a = (free_solver_func *) free_solver_funcs->p;
    for (int i = 0; i < free_solver_funcs->len/sizeof(free_solver_func); i++)
      a[i] ();
    array_free (free_solver_funcs);
  }

  delete (all);
  pfree (all,__func__,__FILE__,__LINE__); all = NULL;
  for (Event * ev = Events; !ev->last; ev++) {
    Event * e = ev->next;
    while (e) {
      Event * next = e->next;
      pfree (e,__func__,__FILE__,__LINE__);
      e = next;
    }
  }

  pfree (Events,__func__,__FILE__,__LINE__); Events = NULL;
  pfree (_attribute,__func__,__FILE__,__LINE__); _attribute = NULL;
  pfree (_constant,__func__,__FILE__,__LINE__); _constant = NULL;
  free_grid();
  qpclose_all();
#if TRACE
  trace_off();
#endif
#if MTRACE
  pmuntrace();
#endif
#if _CADNA
  cadna_end();
#endif
}



void (* boundary_level) (scalar *, int l);
void (* boundary_flux) (vector *);


void boundary (scalar * list)
{ trace ("boundary", "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h", 289);
  if (list == NULL)
    { ; end_trace("boundary", "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h", 291);  return; }
  vector * listf = NULL;
  if (list) for (scalar s = *list, *_i36 = list; ((scalar *)&s)->i >= 0; s = *++_i36)
    if (!is_constant(s) && _attribute[s.i].face)
      listf = vectors_add (listf, _attribute[s.i].v);
  if (listf) {
    boundary_flux (listf);
    pfree (listf,__func__,__FILE__,__LINE__);
  }
  boundary_level (list, -1);
 end_trace("boundary", "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h", 301); }

void cartesian_boundary_level (scalar * list, int l)
{
  { Boundary ** _i = boundaries, * _b; while ((_b = *_i++)) if (_b->level) _b->level (_b, list, l); };
}

void cartesian_boundary_flux (vector * list)
{

}

static double symmetry (Point point, Point neighbor, scalar s)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 314 "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h"

  return val(s,0,0,0);
}

static double antisymmetry (Point point, Point neighbor, scalar s)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 319 "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h"

  return -val(s,0,0,0);
}

double (* default_scalar_bc[]) (Point, Point, scalar) = {
  symmetry, symmetry, symmetry, symmetry, symmetry, symmetry
};

scalar cartesian_init_scalar (scalar s, const char * name)
{

  char * pname;
  if (name) {
    pfree (_attribute[s.i].name,__func__,__FILE__,__LINE__);
    pname = pstrdup (name,__func__,__FILE__,__LINE__);
  }
  else
    pname = _attribute[s.i].name;
  pfree (_attribute[s.i].boundary,__func__,__FILE__,__LINE__);
  pfree (_attribute[s.i].boundary_homogeneous,__func__,__FILE__,__LINE__);

  _attribute[s.i] = (const _Attributes){0};
  _attribute[s.i].name = pname;

  _attribute[s.i].boundary = (double (**)(Point, Point, scalar))
    pmalloc (nboundary*sizeof (void (*)()),__func__,__FILE__,__LINE__);
  _attribute[s.i].boundary_homogeneous = (double (**)(Point, Point, scalar))
    pmalloc (nboundary*sizeof (void (*)()),__func__,__FILE__,__LINE__);
  for (int b = 0; b < nboundary; b++)
    _attribute[s.i].boundary[b] = _attribute[s.i].boundary_homogeneous[b] =
      b < 2*3 ? default_scalar_bc[b] : symmetry;
  _attribute[s.i].gradient = NULL;
  {
#line 351
 {
    _attribute[s.i].d.x = 0;
    _attribute[s.i].v.x.i = -1;
  }
#line 351
 {
    _attribute[s.i].d.y = 0;
    _attribute[s.i].v.y.i = -1;
  }
#line 351
 {
    _attribute[s.i].d.z = 0;
    _attribute[s.i].v.z.i = -1;
  }}
  _attribute[s.i].face = false;
  return s;
}

double (* default_vector_bc[]) (Point, Point, scalar) = {
  antisymmetry, antisymmetry,
  antisymmetry, antisymmetry,
  antisymmetry, antisymmetry
};

vector cartesian_init_vector (vector v, const char * name)
{
  struct { char * x, * y, * z; } ext = {".x", ".y", ".z"};
  {
#line 368
 {
    if (name) {
      char cname[strlen(name) + 3];
      sprintf (cname, "%s%s", name, ext.x);
      init_scalar (v.x, cname);
    }
    else
      init_scalar (v.x, NULL);
    _attribute[v.x.i].v = v;
  }
#line 368
 {
    if (name) {
      char cname[strlen(name) + 3];
      sprintf (cname, "%s%s", name, ext.y);
      init_scalar (v.y, cname);
    }
    else
      init_scalar (v.y, NULL);
    _attribute[v.y.i].v = v;
  }
#line 368
 {
    if (name) {
      char cname[strlen(name) + 3];
      sprintf (cname, "%s%s", name, ext.z);
      init_scalar (v.z, cname);
    }
    else
      init_scalar (v.z, NULL);
    _attribute[v.z.i].v = v;
  }}

  for (int d = 0; d < nboundary; d++)
    _attribute[v.x.i].boundary[d] = _attribute[v.x.i].boundary_homogeneous[d] =
      d < 2*3 ? default_vector_bc[d] : antisymmetry;
  return v;
}

vector cartesian_init_face_vector (vector v, const char * name)
{
  v = cartesian_init_vector (v, name);
  {
#line 388
 {
    _attribute[v.x.i].d.x = -1;
    _attribute[v.x.i].face = true;
  }
#line 388
 {
    _attribute[v.y.i].d.y = -1;
    _attribute[v.y.i].face = true;
  }
#line 388
 {
    _attribute[v.z.i].d.z = -1;
    _attribute[v.z.i].face = true;
  }}
  for (int d = 0; d < nboundary; d++)
    _attribute[v.x.i].boundary[d] = _attribute[v.x.i].boundary_homogeneous[d] = NULL;
  return v;
}

tensor cartesian_init_tensor (tensor t, const char * name)
{
  struct { char * x, * y, * z; } ext = {".x", ".y", ".z"};
  {
#line 400
 {
    if (name) {
      char cname[strlen(name) + 3];
      sprintf (cname, "%s%s", name, ext.x);
      init_vector (t.x, cname);
    }
    else
      init_vector (t.x, NULL);
  }
#line 400
 {
    if (name) {
      char cname[strlen(name) + 3];
      sprintf (cname, "%s%s", name, ext.y);
      init_vector (t.y, cname);
    }
    else
      init_vector (t.y, NULL);
  }
#line 400
 {
    if (name) {
      char cname[strlen(name) + 3];
      sprintf (cname, "%s%s", name, ext.z);
      init_vector (t.z, cname);
    }
    else
      init_vector (t.z, NULL);
  }}
#line 424 "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h"
    assert (false);

  return t;
}

void output_cells (FILE * fp)
{
   { foreach(){

#line 431 "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h"
 {
    Delta /= 2.;
#line 443 "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h"
      for (int i = -1; i <= 1; i += 2) {
 fprintf (fp, "%g %g %g\n%g %g %g\n%g %g %g\n%g %g %g\n%g %g %g\n\n",
   x - Delta, y - Delta, z + i*Delta,
   x - Delta, y + Delta, z + i*Delta,
   x + Delta, y + Delta, z + i*Delta,
   x + Delta, y - Delta, z + i*Delta,
   x - Delta, y - Delta, z + i*Delta);
 for (int j = -1; j <= 1; j += 2)
   fprintf (fp, "%g %g %g\n%g %g %g\n\n",
     x + i*Delta, y + j*Delta, z - Delta,
     x + i*Delta, y + j*Delta, z + Delta);
      }

  } } end_foreach(); }
  fflush (fp);
}

static char * replace_ (const char * vname)
{
  char * name = pstrdup (vname,__func__,__FILE__,__LINE__), * c = name;
  while (*c != '\0') {
    if (*c == '.')
      *c = '_';
    c++;
  }
  return name;
}

static void debug_plot (FILE * fp, const char * name, const char * cells,
   const char * stencil)
{
  char * vname = replace_ (name);
  fprintf (fp,
    "  load 'debug.plot'\n"
    "  v=%s\n"







    "  splot '%s' w l lc 0, "
    "'%s' u 1+4*v:2+4*v:3+4*v:4+4*v w labels tc lt 1"
           " title columnhead(4+4*v)",

    vname, cells, stencil);
  pfree (vname,__func__,__FILE__,__LINE__);
}

void cartesian_debug (Point point)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 494 "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h"

  char name[80] = "cells";
  if (pid() > 0)
    sprintf (name, "cells-%d", pid());
  FILE * fp = fopen (name, "w");
  output_cells (fp);
  fclose (fp);

  char stencil[80] = "stencil";
  if (pid() > 0)
    sprintf (stencil, "stencil-%d", pid());
  fp = fopen (stencil, "w");
  if (all) for (scalar v = *all, *_i37 = all; ((scalar *)&v)->i >= 0; v = *++_i37)





    fprintf (fp, "x y z %s ", _attribute[v.i].name);

  fputc ('\n', fp);
#line 541 "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h"
    for (int k = -2; k <= 2; k++)
      for (int l = -2; l <= 2; l++)
 for (int m = -2; m <= 2; m++) {
   if (all) for (scalar v = *all, *_i38 = all; ((scalar *)&v)->i >= 0; v = *++_i38) {
     fprintf (fp, "%g %g %g ",
       x + k*Delta + _attribute[v.i].d.x*Delta/2.,
       y + l*Delta + _attribute[v.i].d.y*Delta/2.,
       z + m*Delta + _attribute[v.i].d.z*Delta/2.);
     if (allocated(k,l,m))
       fprintf (fp, "%g ", val(v,k,l,m));
     else
       fputs ("n/a ", fp);
   }
   fputc ('\n', fp);
 }

  fclose (fp);

  fp = fopen ("debug.plot", "w");
  fprintf (fp,
    "set term x11\n"
    "set size ratio -1\n"
    "set key outside\n");
  if (all) for (scalar s = *all, *_i39 = all; ((scalar *)&s)->i >= 0; s = *++_i39) {
    char * name = replace_ (_attribute[s.i].name);
    fprintf (fp, "%s = %d\n", name, s.i);
    pfree (name,__func__,__FILE__,__LINE__);
  }
  fclose (fp);

  fprintf (qstderr(), "Last point stencils can be displayed using (in gnuplot)\n");
  debug_plot (qstderr(), _attribute[0].name, name, stencil);
  fflush (qstderr());

  fp = fopen ("plot", "w");
  debug_plot (fp, _attribute[0].name, name, stencil);
  fclose (fp);
}

void cartesian_methods()
{
  init_scalar = cartesian_init_scalar;
  init_vector = cartesian_init_vector;
  init_tensor = cartesian_init_tensor;
  init_face_vector = cartesian_init_face_vector;
  boundary_level = cartesian_boundary_level;
  boundary_flux = cartesian_boundary_flux;
  debug = cartesian_debug;
}

struct _interpolate {
  scalar v;
  double x, y, z;
};

static double interpolate_linear (Point point, struct _interpolate p)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 597 "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h"

  scalar v = p.v;
#line 614 "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h"
  x = (p.x - x)/Delta - _attribute[v.i].d.x/2.;
  y = (p.y - y)/Delta - _attribute[v.i].d.y/2.;
  z = (p.z - z)/Delta - _attribute[v.i].d.z/2.;
  int i = sign(x), j = sign(y), k = sign(z);
  x = fabs(x); y = fabs(y); z = fabs(z);

  return (((val(v,0,0,0)*(1. - x) + val(v,i,0,0)*x)*(1. - y) +
    (val(v,0,j,0)*(1. - x) + val(v,i,j,0)*x)*y)*(1. - z) +
   ((val(v,0,0,k)*(1. - x) + val(v,i,0,k)*x)*(1. - y) +
    (val(v,0,j,k)*(1. - x) + val(v,i,j,k)*x)*y)*z);

}


double interpolate (struct _interpolate p)
{ trace ("interpolate", "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h", 629);
  Point point = locate ((struct _locate){p.x, p.y, p.z});  int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 630 "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h"

  if (point.level < 0)
    { double _ret =  nodata; end_trace("interpolate", "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h", 632);  return _ret; }
  { double _ret =  interpolate_linear (point, p); end_trace("interpolate", "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h", 633);  return _ret; }
 end_trace("interpolate", "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h", 634); }


void interpolate_array (scalar * list, coord * a, int n, double * v, bool linear)
{ trace ("interpolate_array", "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h", 638);
  int j = 0;
  for (int i = 0; i < n; i++) {
    Point point = locate ((struct _locate){a[i].x, a[i].y, a[i].z});  int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 641 "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h"

    if (point.level >= 0) {
      if (list) for (scalar s = *list, *_i40 = list; ((scalar *)&s)->i >= 0; s = *++_i40)
 v[j++] = !linear ? val(s,0,0,0) :
   interpolate_linear (point,
         (struct _interpolate){s, a[i].x, a[i].y, a[i].z});
    }
    else
      if (list) for (scalar s = *list, *_i41 = list; ((scalar *)&s)->i >= 0; s = *++_i41)
 v[j++] = nodata;
  }
#if 1
  if (pid() == 0)
    MPI_Reduce (MPI_IN_PLACE, v, n*list_len(list), MPI_DOUBLE,
  MPI_MIN, 0, MPI_COMM_WORLD);
  else
    MPI_Reduce (v, v, n*list_len(list), MPI_DOUBLE,
  MPI_MIN, 0, MPI_COMM_WORLD);
#endif
 end_trace("interpolate_array", "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h", 660); }



typedef int bid;

bid new_bid()
{
  int b = nboundary++;
  if (all) for (scalar s = *all, *_i42 = all; ((scalar *)&s)->i >= 0; s = *++_i42) {
    _attribute[s.i].boundary = (double (**)(Point, Point, scalar))
      prealloc (_attribute[s.i].boundary, nboundary*sizeof (void (*)()),__func__,__FILE__,__LINE__);
    _attribute[s.i].boundary_homogeneous = (double (**)(Point, Point, scalar))
      prealloc (_attribute[s.i].boundary_homogeneous, nboundary*sizeof (void (*)()),__func__,__FILE__,__LINE__);
  }
  if (all) for (scalar s = *all, *_i43 = all; ((scalar *)&s)->i >= 0; s = *++_i43) {
    if (_attribute[s.i].v.x.i < 0)
      _attribute[s.i].boundary[b] = _attribute[s.i].boundary_homogeneous[b] = symmetry;
    else if (_attribute[s.i].v.x.i == s.i) {
      vector v = _attribute[s.i].v;
      {
#line 680

 _attribute[v.y.i].boundary[b] = _attribute[v.y.i].boundary_homogeneous[b] = symmetry;
#line 680

 _attribute[v.z.i].boundary[b] = _attribute[v.z.i].boundary_homogeneous[b] = symmetry;
#line 680

 _attribute[v.x.i].boundary[b] = _attribute[v.x.i].boundary_homogeneous[b] = symmetry;}
      _attribute[v.x.i].boundary[b] = _attribute[v.x.i].boundary_homogeneous[b] =
 _attribute[v.x.i].face ? NULL : antisymmetry;
    }
  }
  return b;
}



static double periodic_bc (Point point, Point neighbor, scalar s)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 692 "/home/miro3/Documents/Programming/basilisk/src/grid/cartesian-common.h"

  return nodata;
}

static void periodic_boundary (int d)
{

  if (all) for (scalar s = *all, *_i44 = all; ((scalar *)&s)->i >= 0; s = *++_i44)
    _attribute[s.i].boundary[d] = _attribute[s.i].boundary_homogeneous[d] = periodic_bc;

  if (all) for (scalar s = *all, *_i45 = all; ((scalar *)&s)->i >= 0; s = *++_i45)
    if (_attribute[s.i].face) {
      vector v = _attribute[s.i].v;
      _attribute[v.x.i].boundary[d] = _attribute[v.x.i].boundary_homogeneous[d] = NULL;
    }

  default_scalar_bc[d] = periodic_bc;
  default_vector_bc[d] = periodic_bc;
}

int Periods[3];

void periodic (int dir)
{





    assert (dir <= back);


  int c = dir/2;
  periodic_boundary (2*c);
  periodic_boundary (2*c + 1);
  Periods[c] = true;
}
#line 4 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"

#ifndef foreach_level_or_leaf
# define foreach_level_or_leaf foreach_level
# define end_foreach_level_or_leaf end_foreach_level
#endif

#ifndef foreach_coarse_level
# define foreach_coarse_level foreach_level
# define end_foreach_coarse_level end_foreach_level
#endif










void (* restriction) (scalar *);

static inline void restriction_average (Point point, scalar s)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 27 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"

  double sum = 0.;
   { foreach_child()
    sum += val(s,0,0,0); end_foreach_child(); }
  val(s,0,0,0) = sum/(1 << 3);
}

static inline void restriction_volume_average (Point point, scalar s)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 35 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"

if (!is_constant(cm)) {
#undef val_cm
#define val_cm(a,i,j,k) val(a,i,j,k)
#undef fine_cm
#define fine_cm(a,i,j,k) fine(a,i,j,k)
#undef coarse_cm
#define coarse_cm(a,i,j,k) coarse(a,i,j,k)
#line 35

  double sum = 0.;
   { foreach_child()
    sum += val_cm(cm,0,0,0)*val(s,0,0,0); end_foreach_child(); }
  val(s,0,0,0) = sum/(1 << 3)/val_cm(cm,0,0,0);
 }
if (is_constant(cm)) {
const double _const_cm = _constant[cm.i -_NVARMAX];
NOT_UNUSED(_const_cm);
#undef val_cm
#define val_cm(a,i,j,k) _const_cm
#undef fine_cm
#define fine_cm(a,i,j,k) _const_cm
#undef coarse_cm
#define coarse_cm(a,i,j,k) _const_cm
#line 35

  double sum = 0.;
   { foreach_child()
    sum += val_cm(cm,0,0,0)*val(s,0,0,0); end_foreach_child(); }
  val(s,0,0,0) = sum/(1 << 3)/val_cm(cm,0,0,0);
 }}

static inline void face_average (Point point, vector v)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 43 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"

  {
#line 44
 {







      val(v.x,0,0,0) = (fine(v.x,0,0,0) + fine(v.x,0,1,0) +
        fine(v.x,0,0,1) + fine(v.x,0,1,1))/4.;
      val(v.x,1,0,0) = (fine(v.x,2,0,0) + fine(v.x,2,1,0) +
  fine(v.x,2,0,1) + fine(v.x,2,1,1))/4.;

  }
#line 44
 {







      val(v.y,0,0,0) = (fine(v.y,0,0,0) + fine(v.y,0,0,1) +
        fine(v.y,1,0,0) + fine(v.y,1,0,1))/4.;
      val(v.y,0,1,0) = (fine(v.y,0,2,0) + fine(v.y,0,2,1) +
  fine(v.y,1,2,0) + fine(v.y,1,2,1))/4.;

  }
#line 44
 {







      val(v.z,0,0,0) = (fine(v.z,0,0,0) + fine(v.z,1,0,0) +
        fine(v.z,0,1,0) + fine(v.z,1,1,0))/4.;
      val(v.z,0,0,1) = (fine(v.z,0,0,2) + fine(v.z,1,0,2) +
  fine(v.z,0,1,2) + fine(v.z,1,1,2))/4.;

  }}
}

static inline void restriction_face (Point point, scalar s)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 61 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"

  face_average (point, _attribute[s.i].v);
}

static inline void no_restriction (Point point, scalar s) { int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 64 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"
}

static inline void no_data (Point point, scalar s) { int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 67 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"

   { foreach_child()
    val(s,0,0,0) = nodata; end_foreach_child(); }
}

void wavelet (scalar s, scalar w)
{
  restriction (((scalar []){s,{-1}}));
  for (int l = depth() - 1; l >= 0; l--)
     { foreach_coarse_level (l){

#line 76 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"
 {
      double sc[1 << 3];
      int c = 0;
       { foreach_child()
 sc[c++] = val(s,0,0,0); end_foreach_child(); }
      _attribute[s.i].prolongation (point, s);
      c = 0;
       { foreach_child() {

 val(w,0,0,0) = sc[c] - val(s,0,0,0);
 val(s,0,0,0) = sc[c++];
      } end_foreach_child(); }
    } } end_foreach_coarse_level(); }

   { foreach_level(0){

#line 90 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"
 val(w,0,0,0) = 0.; } end_foreach_level(); }
}

static inline double bilinear (Point point, scalar s)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 94 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"








    return (27.*coarse(s,0,0,0) +
     9.*(coarse(s,child.x,0,0) + coarse(s,0,child.y,0) +
  coarse(s,0,0,child.z)) +
     3.*(coarse(s,child.x,child.y,0) + coarse(s,child.x,0,child.z) +
  coarse(s,0,child.y,child.z)) +
     coarse(s,child.x,child.y,child.z))/64.;

}

static inline void refine_bilinear (Point point, scalar s)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 112 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"

   { foreach_child()
    val(s,0,0,0) = bilinear (point, s); end_foreach_child(); }
}

static inline double quadratic (double a, double b, double c)
{
  return (30.*a + 5.*b - 3.*c)/32.;
}

static inline double biquadratic (Point point, scalar s)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 123 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"

#line 138 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"
  assert (false);
  return 0.;

}

static inline double biquadratic_vertex (Point point, scalar s)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 144 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"







  assert (false);
  return 0.;

}

static inline void refine_biquadratic (Point point, scalar s)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 157 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"

   { foreach_child()
    val(s,0,0,0) = biquadratic (point, s); end_foreach_child(); }
}

static inline void refine_linear (Point point, scalar s)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 163 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"

if (!is_constant(cm)) {
#undef val_cm
#define val_cm(a,i,j,k) val(a,i,j,k)
#undef fine_cm
#define fine_cm(a,i,j,k) fine(a,i,j,k)
#undef coarse_cm
#define coarse_cm(a,i,j,k) coarse(a,i,j,k)
#line 163

  coord g;
  if (_attribute[s.i].gradient)
    {
#line 166

      g.x = _attribute[s.i].gradient (val(s,-1,0,0), val(s,0,0,0), val(s,1,0,0));
#line 166

      g.y = _attribute[s.i].gradient (val(s,0,-1,0), val(s,0,0,0), val(s,0,1,0));
#line 166

      g.z = _attribute[s.i].gradient (val(s,0,0,-1), val(s,0,0,0), val(s,0,0,1));}
  else
    {
#line 169

      g.x = (val(s,1,0,0) - val(s,-1,0,0))/2.;
#line 169

      g.y = (val(s,0,1,0) - val(s,0,-1,0))/2.;
#line 169

      g.z = (val(s,0,0,1) - val(s,0,0,-1))/2.;}

  double sc = val(s,0,0,0), cmc = 4.*val_cm(cm,0,0,0), sum = val_cm(cm,0,0,0)*(1 << 3);
   { foreach_child() {
    val(s,0,0,0) = sc;
    {
#line 175

      val(s,0,0,0) += child.x*g.x*val_cm(cm,-child.x,0,0)/cmc;
#line 175

      val(s,0,0,0) += child.y*g.y*val_cm(cm,0,-child.y,0)/cmc;
#line 175

      val(s,0,0,0) += child.z*g.z*val_cm(cm,0,0,-child.z)/cmc;}
    sum -= val_cm(cm,0,0,0);
  } end_foreach_child(); }
  assert (fabs(sum) < 1e-10);
 }
if (is_constant(cm)) {
const double _const_cm = _constant[cm.i -_NVARMAX];
NOT_UNUSED(_const_cm);
#undef val_cm
#define val_cm(a,i,j,k) _const_cm
#undef fine_cm
#define fine_cm(a,i,j,k) _const_cm
#undef coarse_cm
#define coarse_cm(a,i,j,k) _const_cm
#line 163

  coord g;
  if (_attribute[s.i].gradient)
    {
#line 166

      g.x = _attribute[s.i].gradient (val(s,-1,0,0), val(s,0,0,0), val(s,1,0,0));
#line 166

      g.y = _attribute[s.i].gradient (val(s,0,-1,0), val(s,0,0,0), val(s,0,1,0));
#line 166

      g.z = _attribute[s.i].gradient (val(s,0,0,-1), val(s,0,0,0), val(s,0,0,1));}
  else
    {
#line 169

      g.x = (val(s,1,0,0) - val(s,-1,0,0))/2.;
#line 169

      g.y = (val(s,0,1,0) - val(s,0,-1,0))/2.;
#line 169

      g.z = (val(s,0,0,1) - val(s,0,0,-1))/2.;}

  double sc = val(s,0,0,0), cmc = 4.*val_cm(cm,0,0,0), sum = val_cm(cm,0,0,0)*(1 << 3);
   { foreach_child() {
    val(s,0,0,0) = sc;
    {
#line 175

      val(s,0,0,0) += child.x*g.x*val_cm(cm,-child.x,0,0)/cmc;
#line 175

      val(s,0,0,0) += child.y*g.y*val_cm(cm,0,-child.y,0)/cmc;
#line 175

      val(s,0,0,0) += child.z*g.z*val_cm(cm,0,0,-child.z)/cmc;}
    sum -= val_cm(cm,0,0,0);
  } end_foreach_child(); }
  assert (fabs(sum) < 1e-10);
 }}

static inline void refine_reset (Point point, scalar v)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 183 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"

   { foreach_child()
    val(v,0,0,0) = 0.; end_foreach_child(); }
}

static inline void refine_injection (Point point, scalar v)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 189 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"

  double val = val(v,0,0,0);
   { foreach_child()
    val(v,0,0,0) = val; end_foreach_child(); }
}

static scalar multigrid_init_scalar (scalar s, const char * name)
{
  s = cartesian_init_scalar (s, name);
  _attribute[s.i].prolongation = refine_bilinear;
  _attribute[s.i].restriction = restriction_average;
  return s;
}

static vector multigrid_init_face_vector (vector v, const char * name)
{
  v = cartesian_init_face_vector (v, name);
  {
#line 206

    _attribute[v.y.i].restriction = no_restriction;
#line 206

    _attribute[v.z.i].restriction = no_restriction;
#line 206

    _attribute[v.x.i].restriction = no_restriction;}
  _attribute[v.x.i].restriction = restriction_face;
  return v;
}

void multigrid_debug (Point point)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 213 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"

  cartesian_debug (point);

  FILE * plot = fopen ("plot", "a");
  if (point.level > 0) {
    char name[80] = "coarse";
    if (pid() > 0)
      sprintf (name, "coarse-%d", pid());
    FILE * fp = fopen (name, "w");
#line 246 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"
      double xc = x - child.x*Delta/2., yc = y - child.y*Delta/2.;
      double zc = z - child.z*Delta/2.;
      for (int k = 0; k <= 1; k++)
 for (int l = 0; l <= 1; l++)
   for (int m = 0; m <= 1; m++) {
     if (all) for (scalar v = *all, *_i46 = all; ((scalar *)&v)->i >= 0; v = *++_i46)
       fprintf (fp, "%g %g %g %g ",
         xc + k*child.x*Delta*2. + _attribute[v.i].d.x*Delta,
         yc + l*child.y*Delta*2. + _attribute[v.i].d.y*Delta,
         zc + m*child.z*Delta*2. + _attribute[v.i].d.z*Delta,
         coarse(v,k*child.x,l*child.y,m*child.z));
     fputc ('\n', fp);
   }
      fprintf (qstderr(), ", '%s' u 1+4*v:2+4*v:3+4*v:4+4*v w labels tc lt 3 t ''",
        name);
      fprintf (plot, ", '%s' u 1+4*v:2+4*v:3+4*v:4+4*v w labels tc lt 3 t ''",
        name);

    fclose (fp);
  }

  if (is_coarse()) {
    char name[80] = "fine";
    if (pid() > 0)
      sprintf (name, "fine-%d", pid());
    FILE * fp = fopen (name, "w");
#line 303 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"
      double xf = x - Delta/4., yf = y - Delta/4., zf = z - Delta/4.;
      for (int k = -2; k <= 3; k++)
 for (int l = -2; l <= 3; l++)
   for (int m = -2; m <= 3; m++) {
     if (all) for (scalar v = *all, *_i47 = all; ((scalar *)&v)->i >= 0; v = *++_i47) {
       fprintf (fp, "%g %g %g ",
         xf + k*Delta/2. + _attribute[v.i].d.x*Delta/4.,
         yf + l*Delta/2. + _attribute[v.i].d.y*Delta/4.,
         zf + m*Delta/2. + _attribute[v.i].d.z*Delta/4.);
       if (allocated_child(k,l,m))
  fprintf (fp, "%g ", fine(v,k,l,m));
       else
  fputs ("n/a ", fp);
     }
     fputc ('\n', fp);
   }
      fprintf (qstderr(), ", '%s' u 1+4*v:2+4*v:3+4*v:4+4*v w labels tc lt 2 t ''",
        name);
      fprintf (plot, ", '%s' u 1+4*v:2+4*v:3+4*v:4+4*v w labels tc lt 2 t ''",
        name);

    fclose (fp);
  }
  fflush (qstderr());
  fclose (plot);
}

static void multigrid_restriction (scalar * list)
{
  scalar * listdef = NULL, * listc = NULL, * list2 = NULL;
  if (list) for (scalar s = *list, *_i48 = list; ((scalar *)&s)->i >= 0; s = *++_i48)
    if (!is_constant (s)) {
      if (_attribute[s.i].restriction == restriction_average) {
 listdef = list_add (listdef, s);
 list2 = list_add (list2, s);
      }
      else if (_attribute[s.i].restriction != no_restriction) {
 listc = list_add (listc, s);
 if (_attribute[s.i].face)
   {
#line 342

     list2 = list_add (list2, _attribute[s.i].v.x);
#line 342

     list2 = list_add (list2, _attribute[s.i].v.y);
#line 342

     list2 = list_add (list2, _attribute[s.i].v.z);}
 else
   list2 = list_add (list2, s);
      }
    }

  if (listdef || listc) {
    for (int l = depth() - 1; l >= 0; l--) {
       { foreach_coarse_level(l){

#line 351 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"
 {
 if (listdef) for (scalar s = *listdef, *_i49 = listdef; ((scalar *)&s)->i >= 0; s = *++_i49)
   restriction_average (point, s);
 if (listc) for (scalar s = *listc, *_i50 = listc; ((scalar *)&s)->i >= 0; s = *++_i50)
   _attribute[s.i].restriction (point, s);
      } } end_foreach_coarse_level(); }
      { Boundary ** _i = boundaries, * _b; while ((_b = *_i++)) if (_b->level) _b->level (_b, list2, l); };
    }
    pfree (listdef,__func__,__FILE__,__LINE__);
    pfree (listc,__func__,__FILE__,__LINE__);
    pfree (list2,__func__,__FILE__,__LINE__);
  }
}

void multigrid_methods()
{
  cartesian_methods();
  debug = multigrid_debug;
  init_scalar = multigrid_init_scalar;
  init_face_vector = multigrid_init_face_vector;
  restriction = multigrid_restriction;
}







void subtree_size (scalar size, bool leaves)
{




   { foreach(){

#line 386 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"

    val(size,0,0,0) = 1; } end_foreach(); }





  { Boundary ** _i = boundaries, * _b; while ((_b = *_i++)) if (_b->restriction) _b->restriction (_b, ((scalar []){size,{-1}}), depth()); };
  for (int l = depth() - 1; l >= 0; l--) {
     { foreach_coarse_level(l){

#line 395 "/home/miro3/Documents/Programming/basilisk/src/grid/multigrid-common.h"
 {
      double sum = !leaves;
       { foreach_child()
 sum += val(size,0,0,0); end_foreach_child(); }
      val(size,0,0,0) = sum;
    } } end_foreach_coarse_level(); }
    { Boundary ** _i = boundaries, * _b; while ((_b = *_i++)) if (_b->restriction) _b->restriction (_b, ((scalar []){size,{-1}}), l); };
  }
}
#line 5 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"












int refine_cell (Point point, scalar * list, int flag, Cache * refined)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 18 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"

  int nr = 0;


  if (level > 0)
    for (int k = 0; k != 2*child.x; k += child.x)

      for (int l = 0; l != 2*child.y; l += child.y)


 for (int m = 0; m != 2*child.z; m += child.z)

   if (aparent(k,l,m).pid >= 0 && is_leaf(aparent(k,l,m))) {
     Point p = point;


     p.level = point.level - 1;
     p.i = (point.i + 2)/2 + k;

       p.j = (point.j + 2)/2 + l;


       p.k = (point.k + 2)/2 + m;

       nr += refine_cell (p, list, flag, refined);
     aparent(k,l,m).flags |= flag;
   }



  cell.flags &= ~leaf;


  increment_neighbors (point);

  int cflag = is_active(cell) ? (active|leaf) : leaf;
   { foreach_child()
    cell.flags |= cflag; end_foreach_child(); }


  if (list) for (scalar s = *list, *_i51 = list; ((scalar *)&s)->i >= 0; s = *++_i51)
    if (is_local(cell) || _attribute[s.i].face)
      _attribute[s.i].refine (point, s);

#if 1
  if (is_border(cell)) {
     { foreach_child() {
      bool bord = false;
       { foreach_neighbor() {
 if (!is_local(cell) || (level > 0 && !is_local(aparent(0,0,0))))
   bord = true, foreach_neighbor_break();
 if ((!is_leaf (cell) && cell.neighbors && cell.pid >= 0))
    { foreach_child()
     if (!is_local(cell))
       bord = true, foreach_child_break(); end_foreach_child(); }
 if (bord)
   foreach_neighbor_break();
      } end_foreach_neighbor(); }
      if (bord)
 cell.flags |= border;
    } end_foreach_child(); }
    if (refined)
      cache_append (refined, point, cell.flags);
    nr++;
  }
#endif
  return nr;
}





bool coarsen_cell (Point point, scalar * list)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 92 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"




  int pid = cell.pid;
   { foreach_child()
    if (cell.neighbors || (cell.pid < 0 && cell.pid != pid))
      return false; end_foreach_child(); }



  if (list) for (scalar s = *list, *_i52 = list; ((scalar *)&s)->i >= 0; s = *++_i52) {
    _attribute[s.i].restriction (point, s);
    if (_attribute[s.i].coarsen)
      _attribute[s.i].coarsen (point, s);
  }


  cell.flags |= leaf;


  decrement_neighbors (point);

#if 1
  if (!is_local(cell)) {
    cell.flags &= ~(active|border);
     { foreach_neighbor(1)
      if (cell.neighbors)
  { foreach_child()
   if (allocated(0,0,0) && is_local(cell) && !is_border(cell))
     cell.flags |= border; end_foreach_child(); } end_foreach_neighbor(); }
  }
#endif

  return true;
}

void coarsen_cell_recursive (Point point, scalar * list)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 130 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"



   { foreach_child()
    if (cell.neighbors)
       { foreach_neighbor(1)
 if ((!is_leaf (cell) && cell.neighbors && cell.pid >= 0))
   coarsen_cell_recursive (point, list); end_foreach_neighbor(); } end_foreach_child(); }

  assert (coarsen_cell (point, list));
}

void mpi_boundary_refine (scalar *);
void mpi_boundary_coarsen (int, int);
void mpi_boundary_update (scalar *);

typedef struct {
  int nc, nf;
} astats;

struct Adapt {
  scalar * slist;
  double * max;
  int maxlevel;
  int minlevel;
  scalar * list;
};


astats adapt_wavelet (struct Adapt p)
{ trace ("adapt_wavelet", "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h", 160);
  if (p.list == NULL)
    p.list = all;
  if (is_constant(cm))
    restriction (p.slist);
  else {
    scalar * listr = list_concat (((scalar []){cm,{-1}}), p.slist);
    restriction (listr);
    pfree (listr,__func__,__FILE__,__LINE__);
  }

  astats st = {0, 0};
  scalar * listc = NULL;
  if (p.list) for (scalar s = *p.list, *_i53 = p.list; ((scalar *)&s)->i >= 0; s = *++_i53)
    if (!is_constant(s) && _attribute[s.i].restriction != no_restriction)
      listc = list_add (listc, s);


  if (p.minlevel < 1)
    p.minlevel = 1;
  ((Tree *)grid)->refined.n = 0;
  static const int refined = 1 << user, too_fine = 1 << (user + 1);
   { foreach_cell(){

#line 182 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"
 {
    if (is_active(cell)) {
      static const int too_coarse = 1 << (user + 2);
      if (is_leaf (cell)) {
 if (cell.flags & too_coarse) {
   cell.flags &= ~too_coarse;
   refine_cell (point, listc, refined, &((Tree *)grid)->refined);
   st.nf++;
 }
 continue;
      }
      else {
 if (cell.flags & refined) {

   cell.flags &= ~too_coarse;
   continue;
 }

 bool local = is_local(cell);
 if (!local)
    { foreach_child()
     if (is_local(cell))
       local = true, foreach_child_break(); end_foreach_child(); }
 if (local) {
   int i = 0;
   static const int just_fine = 1 << (user + 3);
   if (p.slist) for (scalar s = *p.slist, *_i54 = p.slist; ((scalar *)&s)->i >= 0; s = *++_i54) {
     double max = p.max[i++], sc[1 << 3];
     int c = 0;
      { foreach_child()
       sc[c++] = val(s,0,0,0); end_foreach_child(); }
     _attribute[s.i].prolongation (point, s);
     c = 0;
      { foreach_child() {
       double e = fabs(sc[c] - val(s,0,0,0));
       if (e > max && level < p.maxlevel) {
  cell.flags &= ~too_fine;
  cell.flags |= too_coarse;
       }
       else if ((e <= max/1.5 || level > p.maxlevel) &&
         !(cell.flags & (too_coarse|just_fine))) {
  if (level >= p.minlevel)
    cell.flags |= too_fine;
       }
       else if (!(cell.flags & too_coarse)) {
  cell.flags &= ~too_fine;
  cell.flags |= just_fine;
       }
       val(s,0,0,0) = sc[c++];
     } end_foreach_child(); }
   }
    { foreach_child() {
     cell.flags &= ~just_fine;
     if (!is_leaf(cell)) {
       cell.flags &= ~too_coarse;
       if (level >= p.maxlevel)
  cell.flags |= too_fine;
     }
     else if (!is_active(cell))
       cell.flags &= ~too_coarse;
   } end_foreach_child(); }
 }
      }
    }
    else
      continue;
  } } end_foreach_cell(); }
  mpi_boundary_refine (listc);



  for (int l = depth(); l >= 0; l--) {
     { foreach_cell(){

#line 254 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"

      if (!(cell.pid < 0)) {
 if (level == l) {
   if (!is_leaf(cell)) {
     if (cell.flags & refined)

       cell.flags &= ~(refined|too_fine);
     else if (cell.flags & too_fine) {
       if (is_local(cell) && coarsen_cell (point, listc))
  st.nc++;
       cell.flags &= ~too_fine;
     }
   }
   if (cell.flags & too_fine)
     cell.flags &= ~too_fine;
   else if (level > 0 && (aparent(0,0,0).flags & too_fine))
     aparent(0,0,0).flags &= ~too_fine;
   continue;
 }
 else if (is_leaf(cell))
   continue;
      } } end_foreach_cell(); }
    mpi_boundary_coarsen (l, too_fine);
  }
  pfree (listc,__func__,__FILE__,__LINE__);

  mpi_all_reduce (st.nf, MPI_INT, MPI_SUM);
  mpi_all_reduce (st.nc, MPI_INT, MPI_SUM);
  if (st.nc || st.nf)
    mpi_boundary_update (p.list);

  { astats _ret =  st; end_trace("adapt_wavelet", "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h", 285);  return _ret; }
 end_trace("adapt_wavelet", "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h", 286); }
#line 307 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"
static void refine_level (int depth)
{
  int refined;
  do {
    refined = 0;
    ((Tree *)grid)->refined.n = 0;
     { foreach_leaf(){

#line 313 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"

      if (level < depth) {
 refine_cell (point, NULL, 0, &((Tree *)grid)->refined);
 refined++;
 continue;
      } } end_foreach_leaf(); }
    mpi_all_reduce (refined, MPI_INT, MPI_SUM);
    if (refined) {
      mpi_boundary_refine (NULL);
      mpi_boundary_update (NULL);
    }
  } while (refined);
}
#line 352 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"
static void halo_flux (vector * list)
{
  vector * listv = NULL;
  if (list) for (vector v = *list, *_i55 = list; ((scalar *)&v)->i >= 0; v = *++_i55)
    if (!is_constant(v.x))
      listv = vectors_add (listv, v);

  if (listv) {
    for (int l = depth() - 1; l >= 0; l--)
       { foreach_halo (prolongation, l){

#line 361 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"

 {
#line 362
 {
#line 378 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"
   if ((!is_leaf (neighbor(-1,0,0)) && neighbor(-1,0,0).neighbors && neighbor(-1,0,0).pid >= 0))
     if (listv) for (vector f = *listv, *_i56 = listv; ((scalar *)&f)->i >= 0; f = *++_i56)
       val(f.x,0,0,0) = (fine(f.x,0,0,0) + fine(f.x,0,1,0) +
         fine(f.x,0,0,1) + fine(f.x,0,1,1))/4.;
   if ((!is_leaf (neighbor(1,0,0)) && neighbor(1,0,0).neighbors && neighbor(1,0,0).pid >= 0))
     if (listv) for (vector f = *listv, *_i57 = listv; ((scalar *)&f)->i >= 0; f = *++_i57)
       val(f.x,1,0,0) = (fine(f.x,2,0,0) + fine(f.x,2,1,0) +
   fine(f.x,2,0,1) + fine(f.x,2,1,1))/4.;

      }
#line 362
 {
#line 378 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"
   if ((!is_leaf (neighbor(0,-1,0)) && neighbor(0,-1,0).neighbors && neighbor(0,-1,0).pid >= 0))
     if (listv) for (vector f = *listv, *_i56 = listv; ((scalar *)&f)->i >= 0; f = *++_i56)
       val(f.y,0,0,0) = (fine(f.y,0,0,0) + fine(f.y,0,0,1) +
         fine(f.y,1,0,0) + fine(f.y,1,0,1))/4.;
   if ((!is_leaf (neighbor(0,1,0)) && neighbor(0,1,0).neighbors && neighbor(0,1,0).pid >= 0))
     if (listv) for (vector f = *listv, *_i57 = listv; ((scalar *)&f)->i >= 0; f = *++_i57)
       val(f.y,0,1,0) = (fine(f.y,0,2,0) + fine(f.y,0,2,1) +
   fine(f.y,1,2,0) + fine(f.y,1,2,1))/4.;

      }
#line 362
 {
#line 378 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"
   if ((!is_leaf (neighbor(0,0,-1)) && neighbor(0,0,-1).neighbors && neighbor(0,0,-1).pid >= 0))
     if (listv) for (vector f = *listv, *_i56 = listv; ((scalar *)&f)->i >= 0; f = *++_i56)
       val(f.z,0,0,0) = (fine(f.z,0,0,0) + fine(f.z,1,0,0) +
         fine(f.z,0,1,0) + fine(f.z,1,1,0))/4.;
   if ((!is_leaf (neighbor(0,0,1)) && neighbor(0,0,1).neighbors && neighbor(0,0,1).pid >= 0))
     if (listv) for (vector f = *listv, *_i57 = listv; ((scalar *)&f)->i >= 0; f = *++_i57)
       val(f.z,0,0,1) = (fine(f.z,0,0,2) + fine(f.z,1,0,2) +
   fine(f.z,0,1,2) + fine(f.z,1,1,2))/4.;

      }} } end_foreach_halo(); }
    pfree (listv,__func__,__FILE__,__LINE__);
  }
}



static scalar tree_init_scalar (scalar s, const char * name)
{
  s = multigrid_init_scalar (s, name);
  _attribute[s.i].refine = _attribute[s.i].prolongation;
  return s;
}


#line 401

static void refine_face_x (Point point, scalar s)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 403 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"

  assert (3 > 1);
  vector v = _attribute[s.i].v;
  if (!(!is_leaf (neighbor(-1,0,0)) && neighbor(-1,0,0).neighbors && neighbor(-1,0,0).pid >= 0) &&
      (is_local(cell) || is_local(neighbor(-1,0,0)))) {
    double g1 = (val(v.x,0,+1,0) - val(v.x,0,-1,0))/8.;
    double g2 = (val(v.x,0,0,+1) - val(v.x,0,0,-1))/8.;
    for (int j = 0; j <= 1; j++)
      for (int k = 0; k <= 1; k++)
 fine(v.x,0,j,k) = val(v.x,0,0,0) + (2*j - 1)*g1 + (2*k - 1)*g2;
  }
  if (!(!is_leaf (neighbor(1,0,0)) && neighbor(1,0,0).neighbors && neighbor(1,0,0).pid >= 0) && neighbor(1,0,0).neighbors &&
      (is_local(cell) || is_local(neighbor(1,0,0)))) {
    double g1 = (val(v.x,1,+1,0) - val(v.x,1,-1,0))/8.;
    double g2 = (val(v.x,1,0,+1) - val(v.x,1,0,-1))/8.;
    for (int j = 0; j <= 1; j++)
      for (int k = 0; k <= 1; k++)
 fine(v.x,2,j,k) = val(v.x,1,0,0) + (2*j - 1)*g1 + (2*k - 1)*g2;
  }
  if (is_local(cell)) {
    double g1 = (val(v.x,0,+1,0) + val(v.x,1,+1,0) - val(v.x,0,-1,0) - val(v.x,1,-1,0))/16.;
    double g2 = (val(v.x,0,0,+1) + val(v.x,1,0,+1) - val(v.x,0,0,-1) - val(v.x,1,0,-1))/16.;
    for (int j = 0; j <= 1; j++)
      for (int k = 0; k <= 1; k++)
 fine(v.x,1,j,k) = (val(v.x,0,0,0) + val(v.x,1,0,0))/2. + (2*j - 1)*g1 + (2*k - 1)*g2;
  }
}
#line 401

static void refine_face_y (Point point, scalar s)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 403 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"

  assert (3 > 1);
  vector v = _attribute[s.i].v;
  if (!(!is_leaf (neighbor(0,-1,0)) && neighbor(0,-1,0).neighbors && neighbor(0,-1,0).pid >= 0) &&
      (is_local(cell) || is_local(neighbor(0,-1,0)))) {
    double g1 = (val(v.y,0,0,+1) - val(v.y,0,0,-1))/8.;
    double g2 = (val(v.y,+1,0,0) - val(v.y,-1,0,0))/8.;
    for (int j = 0; j <= 1; j++)
      for (int k = 0; k <= 1; k++)
 fine(v.y,k,0,j) = val(v.y,0,0,0) + (2*j - 1)*g1 + (2*k - 1)*g2;
  }
  if (!(!is_leaf (neighbor(0,1,0)) && neighbor(0,1,0).neighbors && neighbor(0,1,0).pid >= 0) && neighbor(0,1,0).neighbors &&
      (is_local(cell) || is_local(neighbor(0,1,0)))) {
    double g1 = (val(v.y,0,1,+1) - val(v.y,0,1,-1))/8.;
    double g2 = (val(v.y,+1,1,0) - val(v.y,-1,1,0))/8.;
    for (int j = 0; j <= 1; j++)
      for (int k = 0; k <= 1; k++)
 fine(v.y,k,2,j) = val(v.y,0,1,0) + (2*j - 1)*g1 + (2*k - 1)*g2;
  }
  if (is_local(cell)) {
    double g1 = (val(v.y,0,0,+1) + val(v.y,0,1,+1) - val(v.y,0,0,-1) - val(v.y,0,1,-1))/16.;
    double g2 = (val(v.y,+1,0,0) + val(v.y,+1,1,0) - val(v.y,-1,0,0) - val(v.y,-1,1,0))/16.;
    for (int j = 0; j <= 1; j++)
      for (int k = 0; k <= 1; k++)
 fine(v.y,k,1,j) = (val(v.y,0,0,0) + val(v.y,0,1,0))/2. + (2*j - 1)*g1 + (2*k - 1)*g2;
  }
}
#line 401

static void refine_face_z (Point point, scalar s)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 403 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"

  assert (3 > 1);
  vector v = _attribute[s.i].v;
  if (!(!is_leaf (neighbor(0,0,-1)) && neighbor(0,0,-1).neighbors && neighbor(0,0,-1).pid >= 0) &&
      (is_local(cell) || is_local(neighbor(0,0,-1)))) {
    double g1 = (val(v.z,+1,0,0) - val(v.z,-1,0,0))/8.;
    double g2 = (val(v.z,0,+1,0) - val(v.z,0,-1,0))/8.;
    for (int j = 0; j <= 1; j++)
      for (int k = 0; k <= 1; k++)
 fine(v.z,j,k,0) = val(v.z,0,0,0) + (2*j - 1)*g1 + (2*k - 1)*g2;
  }
  if (!(!is_leaf (neighbor(0,0,1)) && neighbor(0,0,1).neighbors && neighbor(0,0,1).pid >= 0) && neighbor(0,0,1).neighbors &&
      (is_local(cell) || is_local(neighbor(0,0,1)))) {
    double g1 = (val(v.z,+1,0,1) - val(v.z,-1,0,1))/8.;
    double g2 = (val(v.z,0,+1,1) - val(v.z,0,-1,1))/8.;
    for (int j = 0; j <= 1; j++)
      for (int k = 0; k <= 1; k++)
 fine(v.z,j,k,2) = val(v.z,0,0,1) + (2*j - 1)*g1 + (2*k - 1)*g2;
  }
  if (is_local(cell)) {
    double g1 = (val(v.z,+1,0,0) + val(v.z,+1,0,1) - val(v.z,-1,0,0) - val(v.z,-1,0,1))/16.;
    double g2 = (val(v.z,0,+1,0) + val(v.z,0,+1,1) - val(v.z,0,-1,0) - val(v.z,0,-1,1))/16.;
    for (int j = 0; j <= 1; j++)
      for (int k = 0; k <= 1; k++)
 fine(v.z,j,k,1) = (val(v.z,0,0,0) + val(v.z,0,0,1))/2. + (2*j - 1)*g1 + (2*k - 1)*g2;
  }
}

void refine_face (Point point, scalar s)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 432 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"

  vector v = _attribute[s.i].v;
  {
#line 434

    _attribute[v.x.i].prolongation (point, v.x);
#line 434

    _attribute[v.y.i].prolongation (point, v.y);
#line 434

    _attribute[v.z.i].prolongation (point, v.z);}
}

void refine_face_solenoidal (Point point, scalar s)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 439 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"

  refine_face (point, s);

  if (is_local(cell)) {

    vector v = _attribute[s.i].v;
    double d[1 << 3], p[1 << 3];
    int i = 0;
     { foreach_child() {
      d[i] = 0.;
      {
#line 449

 d[i] += val(v.x,1,0,0) - val(v.x,0,0,0);
#line 449

 d[i] += val(v.y,0,1,0) - val(v.y,0,0,0);
#line 449

 d[i] += val(v.z,0,0,1) - val(v.z,0,0,0);}
      i++;
    } end_foreach_child(); }
#line 463 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"
    static double m[7][7] = {
      {7./12,5./24,3./8,5./24,3./8,1./4,1./3},
      {5./24,7./12,3./8,5./24,1./4,3./8,1./3},
      {3./8,3./8,3./4,1./4,3./8,3./8,1./2},
      {5./24,5./24,1./4,7./12,3./8,3./8,1./3},
      {3./8,1./4,3./8,3./8,3./4,3./8,1./2},
      {1./4,3./8,3./8,3./8,3./8,3./4,1./2},
      {1./3,1./3,1./2,1./3,1./2,1./2,5./6}
    };
    p[0] = 0.;
    for (int i = 0; i < 7; i++) {
      p[i + 1] = 0.;
      for (int j = 0; j < 7; j++)
 p[i + 1] += m[i][j]*d[j+1];
    }
    for (int k = 0; k <= 1; k++) {
      fine(v.x,1,0,k) += p[4+k] - p[0+k];
      fine(v.x,1,1,k) += p[6+k] - p[2+k];
      fine(v.y,0,1,k) += p[2+k] - p[0+k];
      fine(v.y,1,1,k) += p[6+k] - p[4+k];
    }
    fine(v.z,0,0,1) += p[1] - p[0];
    fine(v.z,0,1,1) += p[3] - p[2];
    fine(v.z,1,0,1) += p[5] - p[4];
    fine(v.z,1,1,1) += p[7] - p[6];

  }

}

vector tree_init_face_vector (vector v, const char * name)
{
  v = cartesian_init_face_vector (v, name);
  {
#line 496

    _attribute[v.x.i].restriction = _attribute[v.x.i].refine = no_restriction;
#line 496

    _attribute[v.y.i].restriction = _attribute[v.y.i].refine = no_restriction;
#line 496

    _attribute[v.z.i].restriction = _attribute[v.z.i].refine = no_restriction;}
  _attribute[v.x.i].restriction = restriction_face;
  _attribute[v.x.i].refine = refine_face;
  {
#line 500

    _attribute[v.x.i].prolongation = refine_face_x;
#line 500

    _attribute[v.y.i].prolongation = refine_face_y;
#line 500

    _attribute[v.z.i].prolongation = refine_face_z;}
  return v;
}

static void tree_boundary_level (scalar * list, int l)
{
  int depth = l < 0 ? depth() : l;

  if (tree_is_full()) {
    { Boundary ** _i = boundaries, * _b; while ((_b = *_i++)) if (_b->level) _b->level (_b, list, depth); };
    return;
  }

  scalar * listdef = NULL, * listc = NULL, * list2 = NULL;
  if (list) for (scalar s = *list, *_i58 = list; ((scalar *)&s)->i >= 0; s = *++_i58)
    if (!is_constant (s)) {
      if (_attribute[s.i].restriction == restriction_average) {
 listdef = list_add (listdef, s);
 list2 = list_add (list2, s);
      }
      else if (_attribute[s.i].restriction != no_restriction) {
 listc = list_add (listc, s);
 if (_attribute[s.i].face)
   {
#line 524

     list2 = list_add (list2, _attribute[s.i].v.x);
#line 524

     list2 = list_add (list2, _attribute[s.i].v.y);
#line 524

     list2 = list_add (list2, _attribute[s.i].v.z);}
 else
   list2 = list_add (list2, s);
      }
    }

  if (listdef || listc) {
    { Boundary ** _i = boundaries, * _b; while ((_b = *_i++)) if (_b->restriction) _b->restriction (_b, list2, depth); };
    for (int l = depth - 1; l >= 0; l--) {
       { foreach_coarse_level(l){

#line 534 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"
 {
 if (listdef) for (scalar s = *listdef, *_i59 = listdef; ((scalar *)&s)->i >= 0; s = *++_i59)
   restriction_average (point, s);
 if (listc) for (scalar s = *listc, *_i60 = listc; ((scalar *)&s)->i >= 0; s = *++_i60)
   _attribute[s.i].restriction (point, s);
      } } end_foreach_coarse_level(); }
      { Boundary ** _i = boundaries, * _b; while ((_b = *_i++)) if (_b->restriction) _b->restriction (_b, list2, l); };
    }
    pfree (listdef,__func__,__FILE__,__LINE__);
    pfree (listc,__func__,__FILE__,__LINE__);
    pfree (list2,__func__,__FILE__,__LINE__);
  }

  scalar * listr = NULL;
  vector * listf = NULL;
  if (list) for (scalar s = *list, *_i61 = list; ((scalar *)&s)->i >= 0; s = *++_i61)
    if (!is_constant (s) && _attribute[s.i].refine != no_restriction) {
      if (_attribute[s.i].face)
 listf = vectors_add (listf, _attribute[s.i].v);
      else
 listr = list_add (listr, s);
    }

  if (listr || listf) {
    { Boundary ** _i = boundaries, * _b; while ((_b = *_i++)) if (_b->level) _b->level (_b, list, 0); };
    for (int i = 0; i < depth; i++) {
       { foreach_halo (prolongation, i){

#line 560 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"
 {
 if (listr) for (scalar s = *listr, *_i62 = listr; ((scalar *)&s)->i >= 0; s = *++_i62)
          _attribute[s.i].prolongation (point, s);
 if (listf) for (vector v = *listf, *_i63 = listf; ((scalar *)&v)->i >= 0; v = *++_i63)
   {
#line 564

     _attribute[v.x.i].prolongation (point, v.x);
#line 564

     _attribute[v.y.i].prolongation (point, v.y);
#line 564

     _attribute[v.z.i].prolongation (point, v.z);}
      } } end_foreach_halo(); }
      { Boundary ** _i = boundaries, * _b; while ((_b = *_i++)) if (_b->level) _b->level (_b, list, i + 1); };
    }
    pfree (listr,__func__,__FILE__,__LINE__);
    pfree (listf,__func__,__FILE__,__LINE__);
  }
}

double treex (Point point) { int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 574 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"

  if (level == 0)
    return 0;




  assert (false);
  double i = 0;

  return treex(parent) + i/(1 << 2*(level - 1));
}

double treey (Point point) { int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 587 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"

  if (level == 0)
    return 0;
  return treey(parent) + 4./(1 << 2*(level - 1));
}

void output_tree (FILE * fp)
{
   { foreach_cell(){

#line 595 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"

    if (cell.neighbors)
       { foreach_child()
 if (is_local(cell))
   fprintf (fp, "%g %g\n%g %g\n\n",
     treex(parent), treey(parent), treex(point), treey(point)); end_foreach_child(); }; } end_foreach_cell(); }
}

void tree_check()
{


  long nleaves = 0, nactive = 0;
   { foreach_cell_all(){

#line 608 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"
 {
    if (is_leaf(cell)) {
      assert (cell.pid >= 0);
      nleaves++;
    }
    if (is_local(cell))
      assert (is_active(cell) || (!is_leaf(cell) && !cell.neighbors && cell.pid >= 0));
    if (is_active(cell))
      nactive++;

    int neighbors = 0;
     { foreach_neighbor(1)
      if (allocated(0,0,0) && (!is_leaf (cell) && cell.neighbors && cell.pid >= 0))
 neighbors++; end_foreach_neighbor(); }
    assert (cell.neighbors == neighbors);


    if (!cell.neighbors)
      assert (!allocated_child(0,0,0));
  } } end_foreach_cell_all(); }


  long reachable = 0;
   { foreach_cell(){

#line 631 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"
 {
    if (is_active(cell))
      reachable++;
    else
      continue;
  } } end_foreach_cell(); }
  assert (nactive == reachable);


  reachable = 0;
   { foreach_cell(){

#line 641 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-common.h"

    if (is_leaf(cell)) {
      reachable++;
      continue;
    } } end_foreach_cell(); }
  assert (nleaves == reachable);
}

static void tree_restriction (scalar * list) {
  if (tree_is_full())
    multigrid_restriction (list);

}

void tree_methods()
{
  multigrid_methods();
  init_scalar = tree_init_scalar;
  init_face_vector = tree_init_face_vector;
  boundary_level = tree_boundary_level;
  boundary_flux = halo_flux;
  restriction = tree_restriction;
}
#line 1741 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"

#if 1
#line 1 "grid/tree-mpi.h"
#line 1 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"

int debug_iteration = -1;

void debug_mpi (FILE * fp1);

typedef struct {
  CacheLevel * halo;
  void * buf;
  MPI_Request r;
  int depth;
  int pid;
  int maxdepth;
} Rcv;

typedef struct {
  Rcv * rcv;
  char * name;
  int npid;
} RcvPid;

typedef struct {
  RcvPid * rcv, * snd;
} SndRcv;

typedef struct {
  Boundary parent;

  SndRcv mpi_level, mpi_level_root, restriction;
  Array * send, * receive;
} MpiBoundary;

static void cache_level_init (CacheLevel * c)
{
  c->p = NULL;
  c->n = c->nm = 0;
}

static void rcv_append (Point point, Rcv * rcv)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 39 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"

  if (level > rcv->depth) {
    rcv->halo = (CacheLevel *) prealloc (rcv->halo, (level + 1)*sizeof(CacheLevel),__func__,__FILE__,__LINE__);
    for (int j = rcv->depth + 1; j <= level; j++)
      cache_level_init (&rcv->halo[j]);
    rcv->depth = level;
  }
  cache_level_append (&rcv->halo[level], point);
  if (level > rcv->maxdepth)
    rcv->maxdepth = level;
}

void rcv_print (Rcv * rcv, FILE * fp, const char * prefix)
{
  for (int l = 0; l <= rcv->depth; l++)
    if (rcv->halo[l].n > 0)
       { foreach_cache_level(rcv->halo[l], l){

#line 55 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"

 fprintf (fp, "%s%g %g %g %d %d\n", prefix, x, y, z, rcv->pid, level); } end_foreach_cache_level(); }
}

static void rcv_free_buf (Rcv * rcv)
{
  if (rcv->buf) {
    prof_start ("rcv_pid_receive");
    MPI_Wait (&rcv->r, MPI_STATUS_IGNORE);
    pfree (rcv->buf,__func__,__FILE__,__LINE__);
    rcv->buf = NULL;
    prof_stop();
  }
}

static void rcv_destroy (Rcv * rcv)
{
  rcv_free_buf (rcv);
  for (int i = 0; i <= rcv->depth; i++)
    if (rcv->halo[i].n > 0)
      pfree (rcv->halo[i].p,__func__,__FILE__,__LINE__);
  pfree (rcv->halo,__func__,__FILE__,__LINE__);
}

static RcvPid * rcv_pid_new (const char * name)
{
  RcvPid * r = ((RcvPid *) pcalloc (1, sizeof(RcvPid),__func__,__FILE__,__LINE__));
  r->name = pstrdup (name,__func__,__FILE__,__LINE__);
  return r;
}

static Rcv * rcv_pid_pointer (RcvPid * p, int pid)
{
  assert (pid >= 0 && pid < npe());

  int i;
  for (i = 0; i < p->npid; i++)
    if (pid == p->rcv[i].pid)
      break;

  if (i == p->npid) {
    p->rcv = (Rcv *) prealloc (p->rcv, (++p->npid)*sizeof(Rcv),__func__,__FILE__,__LINE__);
    Rcv * rcv = &p->rcv[p->npid-1];
    rcv->pid = pid;
    rcv->depth = rcv->maxdepth = 0;
    rcv->halo = ((CacheLevel *) pmalloc ((1)*sizeof(CacheLevel),__func__,__FILE__,__LINE__));
    rcv->buf = NULL;
    cache_level_init (&rcv->halo[0]);
  }
  return &p->rcv[i];
}

static void rcv_pid_append (RcvPid * p, int pid, Point point)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 108 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"

  rcv_append (point, rcv_pid_pointer (p, pid));
}

static void rcv_pid_append_pids (RcvPid * p, Array * pids)
{

  for (int i = 0; i < p->npid; i++) {
    int pid = p->rcv[i].pid, j, * a;
    for (j = 0, a = pids->p; j < pids->len/sizeof(int); j++,a++)
      if (*a == pid)
 break;
    if (j == pids->len/sizeof(int))
      array_append (pids, &pid, sizeof(int));
  }
}

void rcv_pid_write (RcvPid * p, const char * name)
{
  for (int i = 0; i < p->npid; i++) {
    Rcv * rcv = &p->rcv[i];
    char fname[80];
    sprintf (fname, "%s-%d-%d", name, pid(), rcv->pid);
    FILE * fp = fopen (fname, "w");
    rcv_print (rcv, fp, "");
    fclose (fp);
  }
}

static void rcv_pid_print (RcvPid * p, FILE * fp, const char * prefix)
{
  for (int i = 0; i < p->npid; i++)
    rcv_print (&p->rcv[i], fp, prefix);
}

static void rcv_pid_destroy (RcvPid * p)
{
  for (int i = 0; i < p->npid; i++)
    rcv_destroy (&p->rcv[i]);
  pfree (p->rcv,__func__,__FILE__,__LINE__);
  pfree (p->name,__func__,__FILE__,__LINE__);
  pfree (p,__func__,__FILE__,__LINE__);
}

static Boundary * mpi_boundary = NULL;






void debug_mpi (FILE * fp1);

static void apply_bc (Rcv * rcv, scalar * list, vector * listf, int l,
        MPI_Status s)
{
  double * b = rcv->buf;
   { foreach_cache_level(rcv->halo[l], l){

#line 165 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"
 {
    if (list) for (scalar s = *list, *_i64 = list; ((scalar *)&s)->i >= 0; s = *++_i64)
      val(s,0,0,0) = *b++;
    if (listf) for (vector v = *listf, *_i65 = listf; ((scalar *)&v)->i >= 0; v = *++_i65) {
      {
#line 169
 {
 val(v.x,0,0,0) = *b++;
 if (allocated(1,0,0))
   val(v.x,1,0,0) = *b;
 b++;
      }
#line 169
 {
 val(v.y,0,0,0) = *b++;
 if (allocated(0,1,0))
   val(v.y,0,1,0) = *b;
 b++;
      }
#line 169
 {
 val(v.z,0,0,0) = *b++;
 if (allocated(0,0,1))
   val(v.z,0,0,1) = *b;
 b++;
      }}
    }
  } } end_foreach_cache_level(); }
  size_t size = b - (double *) rcv->buf;
  pfree (rcv->buf,__func__,__FILE__,__LINE__);
  rcv->buf = NULL;

  int rlen;
  MPI_Get_count (&s, MPI_DOUBLE, &rlen);
  if (rlen != size) {
    fprintf (qstderr(),
      "rlen (%d) != size (%ld), %d receiving from %d at level %d\n"
      "Calling debug_mpi(NULL)...\n"
      "Aborting...\n",
      rlen, size, pid(), rcv->pid, l);
    fflush (qstderr());
    debug_mpi (NULL);
    MPI_Abort (MPI_COMM_WORLD, -2);
  }
}
#line 215 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"
static void mpi_recv_check (void * buf, int count, MPI_Datatype datatype,
       int source, int tag,
       MPI_Comm comm, MPI_Status * status,
       const char * name)
{
#line 250 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"
  int errorcode = MPI_Recv (buf, count, datatype, source, tag, comm, status);
  if (errorcode != MPI_SUCCESS) {
    char string[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string (errorcode, string, &resultlen);
    fprintf (qstderr(),
      "ERROR MPI_Recv \"%s\" (count = %d, source = %d, tag = %d):\n%s\n"
      "Calling debug_mpi(NULL)...\n"
      "Aborting...\n",
      name, count, source, tag, string);
    fflush (qstderr());
    debug_mpi (NULL);
    MPI_Abort (MPI_COMM_WORLD, -1);
  }





}

static void rcv_pid_receive (RcvPid * m, scalar * list, vector * listf, int l)
{
  if (m->npid == 0)
    return;

  prof_start ("rcv_pid_receive");

  int len = list_len (list) + 2*3*vectors_len (listf);

  MPI_Request r[m->npid];
  Rcv * rrcv[m->npid];
  int nr = 0;
  for (int i = 0; i < m->npid; i++) {
    Rcv * rcv = &m->rcv[i];
    if (l <= rcv->depth && rcv->halo[l].n > 0) {
      assert (!rcv->buf);
      rcv->buf = pmalloc (sizeof (double)*rcv->halo[l].n*len,__func__,__FILE__,__LINE__);






      MPI_Irecv (rcv->buf, rcv->halo[l].n*len, MPI_DOUBLE, rcv->pid,
   (l), MPI_COMM_WORLD, &r[nr]);
      rrcv[nr++] = rcv;






    }
  }


  if (nr > 0) {
    int i;
    MPI_Status s;
    MPI_Waitany (nr, r, &i, &s);
    while (i != MPI_UNDEFINED) {
      Rcv * rcv = rrcv[i];
      assert (l <= rcv->depth && rcv->halo[l].n > 0);
      assert (rcv->buf);
      apply_bc (rcv, list, listf, l, s);
      MPI_Waitany (nr, r, &i, &s);
    }
  }

  prof_stop();
}

static void rcv_pid_wait (RcvPid * m)
{

  for (int i = 0; i < m->npid; i++)
    rcv_free_buf (&m->rcv[i]);
}

static void rcv_pid_send (RcvPid * m, scalar * list, vector * listf, int l)
{
  if (m->npid == 0)
    return;

  prof_start ("rcv_pid_send");

  int len = list_len (list) + 2*3*vectors_len (listf);


  for (int i = 0; i < m->npid; i++) {
    Rcv * rcv = &m->rcv[i];
    if (l <= rcv->depth && rcv->halo[l].n > 0) {
      assert (!rcv->buf);
      rcv->buf = pmalloc (sizeof (double)*rcv->halo[l].n*len,__func__,__FILE__,__LINE__);
      double * b = rcv->buf;
       { foreach_cache_level(rcv->halo[l], l){

#line 346 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"
 {
 if (list) for (scalar s = *list, *_i66 = list; ((scalar *)&s)->i >= 0; s = *++_i66)
   *b++ = val(s,0,0,0);
 if (listf) for (vector v = *listf, *_i67 = listf; ((scalar *)&v)->i >= 0; v = *++_i67)
   {
#line 350
 {
     *b++ = val(v.x,0,0,0);
     *b++ = allocated(1,0,0) ? val(v.x,1,0,0) : undefined;
   }
#line 350
 {
     *b++ = val(v.y,0,0,0);
     *b++ = allocated(0,1,0) ? val(v.y,0,1,0) : undefined;
   }
#line 350
 {
     *b++ = val(v.z,0,0,0);
     *b++ = allocated(0,0,1) ? val(v.z,0,0,1) : undefined;
   }}
      } } end_foreach_cache_level(); }





      MPI_Isend (rcv->buf, (b - (double *) rcv->buf),
   MPI_DOUBLE, rcv->pid, (l), MPI_COMM_WORLD,
   &rcv->r);
    }
  }

  prof_stop();
}

static void rcv_pid_sync (SndRcv * m, scalar * list, int l)
{
  scalar * listr = NULL;
  vector * listf = NULL;
  if (list) for (scalar s = *list, *_i68 = list; ((scalar *)&s)->i >= 0; s = *++_i68)
    if (!is_constant(s)) {
      if (_attribute[s.i].face)
 listf = vectors_add (listf, _attribute[s.i].v);
      else
 listr = list_add (listr, s);
    }
  rcv_pid_send (m->snd, listr, listf, l);
  rcv_pid_receive (m->rcv, listr, listf, l);
  rcv_pid_wait (m->snd);
  pfree (listr,__func__,__FILE__,__LINE__);
  pfree (listf,__func__,__FILE__,__LINE__);
}

static void snd_rcv_destroy (SndRcv * m)
{
  rcv_pid_destroy (m->rcv);
  rcv_pid_destroy (m->snd);
}

static void snd_rcv_init (SndRcv * m, const char * name)
{
  char s[strlen(name) + 5];
  strcpy (s, name);
  strcat (s, ".rcv");
  m->rcv = rcv_pid_new (s);
  strcpy (s, name);
  strcat (s, ".snd");
  m->snd = rcv_pid_new (s);
}

static void mpi_boundary_destroy (Boundary * b)
{
  MpiBoundary * m = (MpiBoundary *) b;
  snd_rcv_destroy (&m->mpi_level);
  snd_rcv_destroy (&m->mpi_level_root);
  snd_rcv_destroy (&m->restriction);
  array_free (m->send);
  array_free (m->receive);
  pfree (m,__func__,__FILE__,__LINE__);
}


static void mpi_boundary_level (const Boundary * b, scalar * list, int l)
{ trace ("mpi_boundary_level", "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h", 417);
  MpiBoundary * m = (MpiBoundary *) b;
  rcv_pid_sync (&m->mpi_level, list, l);
  rcv_pid_sync (&m->mpi_level_root, list, l);
 end_trace("mpi_boundary_level", "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h", 421); }


static void mpi_boundary_restriction (const Boundary * b, scalar * list, int l)
{ trace ("mpi_boundary_restriction", "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h", 425);
  MpiBoundary * m = (MpiBoundary *) b;
  rcv_pid_sync (&m->restriction, list, l);
 end_trace("mpi_boundary_restriction", "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h", 428); }

void mpi_boundary_new()
{
  mpi_boundary = (Boundary *) ((MpiBoundary *) pcalloc (1, sizeof(MpiBoundary),__func__,__FILE__,__LINE__));
  mpi_boundary->destroy = mpi_boundary_destroy;
  mpi_boundary->level = mpi_boundary_level;
  mpi_boundary->restriction = mpi_boundary_restriction;
  MpiBoundary * mpi = (MpiBoundary *) mpi_boundary;
  snd_rcv_init (&mpi->mpi_level, "mpi_level");
  snd_rcv_init (&mpi->mpi_level_root, "mpi_level_root");
  snd_rcv_init (&mpi->restriction, "restriction");
  mpi->send = array_new();
  mpi->receive = array_new();
  add_boundary (mpi_boundary);
}

static FILE * fopen_prefix (FILE * fp, const char * name, char * prefix)
{
  if (fp) {
    sprintf (prefix, "%s-%d ", name, pid());
    return fp;
  }
  else {
    strcpy (prefix, "");
    char fname[80];
    if (debug_iteration >= 0)
      sprintf (fname, "%s-%d-%d", name, debug_iteration, pid());
    else
      sprintf (fname, "%s-%d", name, pid());
    return fopen (fname, "w");
  }
}

void debug_mpi (FILE * fp1)
{
  void output_cells (FILE * fp);

  char prefix[80];
  FILE * fp;


  if (fp1 == NULL) {
    char name[80];
    sprintf (name, "halo-%d", pid()); remove (name);
    sprintf (name, "cells-%d", pid()); remove (name);
    sprintf (name, "faces-%d", pid()); remove (name);
    sprintf (name, "neighbors-%d", pid()); remove (name);
    sprintf (name, "mpi-level-rcv-%d", pid()); remove (name);
    sprintf (name, "mpi-level-snd-%d", pid()); remove (name);
    sprintf (name, "mpi-level-root-rcv-%d", pid()); remove (name);
    sprintf (name, "mpi-level-root-snd-%d", pid()); remove (name);
    sprintf (name, "mpi-restriction-rcv-%d", pid()); remove (name);
    sprintf (name, "mpi-restriction-snd-%d", pid()); remove (name);
    sprintf (name, "mpi-border-%d", pid()); remove (name);
    sprintf (name, "exterior-%d", pid()); remove (name);
    sprintf (name, "depth-%d", pid()); remove (name);
    sprintf (name, "refined-%d", pid()); remove (name);
  }


  fp = fopen_prefix (fp1, "halo", prefix);
  for (int l = 0; l < depth(); l++)
     { foreach_halo (prolongation, l){

#line 491 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"

       { foreach_child()
      fprintf (fp, "%s%g %g %g %d\n", prefix, x, y, z, level); end_foreach_child(); }; } end_foreach_halo(); }
  if (!fp1)
    fclose (fp);

  if (!fp1) {
    fp = fopen_prefix (fp1, "cells", prefix);
    output_cells (fp);
    fclose (fp);
  }

  fp = fopen_prefix (fp1, "faces", prefix);
   { foreach_face_generic() { int ig = -1; VARIABLES;  if (is_face_x()) {
#line 504
{

#line 504 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"

    fprintf (fp, "%s%g %g %g %d\n", prefix, x, y, z, level); }  }}  { int jg = -1; VARIABLES;  if (is_face_y()) {
#line 504
{

#line 504 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"

    fprintf (fp, "%s%g %g %g %d\n", prefix, x, y, z, level); }  }}  { int kg = -1; VARIABLES;  if (is_face_z()) {
#line 504
{

#line 504 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"

    fprintf (fp, "%s%g %g %g %d\n", prefix, x, y, z, level); }  }}  end_foreach_face_generic()
#line 505
 end_foreach_face(); }
  if (!fp1)
    fclose (fp);

  fp = fopen_prefix (fp1, "neighbors", prefix);
   { foreach(){

#line 510 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"
 {
    int n = 0;
     { foreach_neighbor(1)
      if ((!is_leaf (cell) && cell.neighbors && cell.pid >= 0))
 n++; end_foreach_neighbor(); }
    fprintf (fp, "%s%g %g %g %d\n", prefix, x, y, z, cell.neighbors);
    assert (cell.neighbors == n);
  } } end_foreach(); }
  if (!fp1)
    fclose (fp);

  MpiBoundary * mpi = (MpiBoundary *) mpi_boundary;

  fp = fopen_prefix (fp1, "mpi-level-rcv", prefix);
  rcv_pid_print (mpi->mpi_level.rcv, fp, prefix);
  if (!fp1)
    fclose (fp);

  fp = fopen_prefix (fp1, "mpi-level-root-rcv", prefix);
  rcv_pid_print (mpi->mpi_level_root.rcv, fp, prefix);
  if (!fp1)
    fclose (fp);

  fp = fopen_prefix (fp1, "mpi-restriction-rcv", prefix);
  rcv_pid_print (mpi->restriction.rcv, fp, prefix);
  if (!fp1)
    fclose (fp);

  fp = fopen_prefix (fp1, "mpi-level-snd", prefix);
  rcv_pid_print (mpi->mpi_level.snd, fp, prefix);
  if (!fp1)
    fclose (fp);

  fp = fopen_prefix (fp1, "mpi-level-root-snd", prefix);
  rcv_pid_print (mpi->mpi_level_root.snd, fp, prefix);
  if (!fp1)
    fclose (fp);

  fp = fopen_prefix (fp1, "mpi-restriction-snd", prefix);
  rcv_pid_print (mpi->restriction.snd, fp, prefix);
  if (!fp1)
    fclose (fp);

  fp = fopen_prefix (fp1, "mpi-border", prefix);
   { foreach_cell(){

#line 554 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"
 {
    if (is_border(cell))
      fprintf (fp, "%s%g %g %g %d %d %d\n",
        prefix, x, y, z, level, cell.neighbors, cell.pid);
    else
      continue;
    if (is_leaf(cell))
      continue;
  } } end_foreach_cell(); }
  if (!fp1)
    fclose (fp);

  fp = fopen_prefix (fp1, "exterior", prefix);
   { foreach_cell(){

#line 567 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"
 {
    if (!is_local(cell))
      fprintf (fp, "%s%g %g %g %d %d %d %d\n",
        prefix, x, y, z, level, cell.neighbors,
        cell.pid, cell.flags & leaf);






  } } end_foreach_cell(); }
  if (!fp1)
    fclose (fp);

  fp = fopen_prefix (fp1, "depth", prefix);
  fprintf (fp, "depth: %d %d\n", pid(), depth());
  fprintf (fp, "======= mpi_level.snd ======\n");
  RcvPid * snd = mpi->mpi_level.snd;
  for (int i = 0; i < snd->npid; i++)
    fprintf (fp, "%d %d %d\n", pid(), snd->rcv[i].pid, snd->rcv[i].maxdepth);
  fprintf (fp, "======= mpi_level.rcv ======\n");
  snd = mpi->mpi_level.rcv;
  for (int i = 0; i < snd->npid; i++)
    fprintf (fp, "%d %d %d\n", pid(), snd->rcv[i].pid, snd->rcv[i].maxdepth);
  if (!fp1)
    fclose (fp);

  fp = fopen_prefix (fp1, "refined", prefix);
   { foreach_cache (((Tree *)grid)->refined){

#line 596 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"

    fprintf (fp, "%s%g %g %g %d\n", prefix, x, y, z, level); } end_foreach_cache(); }
  if (!fp1)
    fclose (fp);
}

static void snd_rcv_free (SndRcv * p)
{
  char name[strlen(p->rcv->name) + 1];
  strcpy (name, p->rcv->name);
  rcv_pid_destroy (p->rcv);
  p->rcv = rcv_pid_new (name);
  strcpy (name, p->snd->name);
  rcv_pid_destroy (p->snd);
  p->snd = rcv_pid_new (name);
}

static bool is_root (Point point)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 614 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"

  if ((!is_leaf (cell) && cell.neighbors && cell.pid >= 0))
     { foreach_child()
      if (is_local(cell))
 return true; end_foreach_child(); }
  return false;
}


static bool is_local_prolongation (Point point, Point p)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 624 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"




  struct { int x, y, z; } dp = {p.i - point.i, p.j - point.j, p.k - point.k};

  {
#line 630
 {
    if (dp.x == 0 && ((!is_leaf (neighbor(-1,0,0)) && neighbor(-1,0,0).neighbors && neighbor(-1,0,0).pid >= 0) || (!is_leaf (neighbor(1,0,0)) && neighbor(1,0,0).neighbors && neighbor(1,0,0).pid >= 0)))
      return true;
    if ((!is_leaf (neighbor(dp.x,0,0)) && neighbor(dp.x,0,0).neighbors && neighbor(dp.x,0,0).pid >= 0))
      return true;
  }
#line 630
 {
    if (dp.y == 0 && ((!is_leaf (neighbor(0,-1,0)) && neighbor(0,-1,0).neighbors && neighbor(0,-1,0).pid >= 0) || (!is_leaf (neighbor(0,1,0)) && neighbor(0,1,0).neighbors && neighbor(0,1,0).pid >= 0)))
      return true;
    if ((!is_leaf (neighbor(0,dp.y,0)) && neighbor(0,dp.y,0).neighbors && neighbor(0,dp.y,0).pid >= 0))
      return true;
  }
#line 630
 {
    if (dp.z == 0 && ((!is_leaf (neighbor(0,0,-1)) && neighbor(0,0,-1).neighbors && neighbor(0,0,-1).pid >= 0) || (!is_leaf (neighbor(0,0,1)) && neighbor(0,0,1).neighbors && neighbor(0,0,1).pid >= 0)))
      return true;
    if ((!is_leaf (neighbor(0,0,dp.z)) && neighbor(0,0,dp.z).neighbors && neighbor(0,0,dp.z).pid >= 0))
      return true;
  }}
  return false;
}



static void append_pid (Array * pids, int pid)
{
  for (int i = 0, * p = (int *) pids->p; i < pids->len/sizeof(int); i++, p++)
    if (*p == pid)
      return;
  array_append (pids, &pid, sizeof(int));
}

static int locals_pids (Point point, Array * pids)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 650 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"

  if (is_leaf(cell)) {
    if (is_local(cell)) {
      Point p = point;
       { foreach_neighbor(1) {
 if ((cell.pid >= 0 && cell.pid != pid()) &&
     ((!is_leaf (cell) && cell.neighbors && cell.pid >= 0) || is_local_prolongation (point, p)))
   append_pid (pids, cell.pid);
 if ((!is_leaf (cell) && cell.neighbors && cell.pid >= 0))
    { foreach_child()
     if ((cell.pid >= 0 && cell.pid != pid()))
       append_pid (pids, cell.pid); end_foreach_child(); }
      } end_foreach_neighbor(); }
    }
  }
  else
     { foreach_neighbor(1) {
      if ((cell.pid >= 0 && cell.pid != pid()))
 append_pid (pids, cell.pid);
      if ((!is_leaf (cell) && cell.neighbors && cell.pid >= 0))
  { foreach_child()
   if ((cell.pid >= 0 && cell.pid != pid()))
     append_pid (pids, cell.pid); end_foreach_child(); }
    } end_foreach_neighbor(); }
  return pids->len/sizeof(int);
}

static int root_pids (Point point, Array * pids)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 678 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"

   { foreach_child()
    if ((cell.pid >= 0 && cell.pid != pid()))
      append_pid (pids, cell.pid); end_foreach_child(); }
  return pids->len/sizeof(int);
}







static void rcv_pid_row (RcvPid * m, int l, int * row)
{
  for (int i = 0; i < npe(); i++)
    row[i] = 0;
  for (int i = 0; i < m->npid; i++) {
    Rcv * rcv = &m->rcv[i];
    if (l <= rcv->depth && rcv->halo[l].n > 0)
      row[rcv->pid] = rcv->halo[l].n;
  }
}

void check_snd_rcv_matrix (SndRcv * sndrcv, const char * name)
{
  int maxlevel = depth();
  mpi_all_reduce (maxlevel, MPI_INT, MPI_MAX);
  int * row = ((int *) pmalloc ((npe())*sizeof(int),__func__,__FILE__,__LINE__));
  for (int l = 0; l <= maxlevel; l++) {
    int status = 0;
    if (pid() == 0) {


      int ** send = matrix_new (npe(), npe(), sizeof(int));
      int ** receive = matrix_new (npe(), npe(), sizeof(int));
      rcv_pid_row (sndrcv->snd, l, row);
      MPI_Gather (row, npe(), MPI_INT, &send[0][0], npe(), MPI_INT, 0,
    MPI_COMM_WORLD);
      rcv_pid_row (sndrcv->rcv, l, row);
      MPI_Gather (row, npe(), MPI_INT, &receive[0][0], npe(), MPI_INT, 0,
    MPI_COMM_WORLD);

      int * astatus = ((int *) pmalloc ((npe())*sizeof(int),__func__,__FILE__,__LINE__));
      for (int i = 0; i < npe(); i++)
 astatus[i] = 0;
      for (int i = 0; i < npe(); i++)
 for (int j = 0; j < npe(); j++)
   if (send[i][j] != receive[j][i]) {
     fprintf (qstderr(), "%s: %d sends    %d to   %d at level %d\n",
       name, i, send[i][j], j, l);
     fprintf (qstderr(), "%s: %d receives %d from %d at level %d\n",
       name, j, receive[j][i], i, l);
     fflush (qstderr());
     for (int k = i - 2; k <= i + 2; k++)
       if (k >= 0 && k < npe())
  astatus[k] = 1;
     for (int k = j - 2; k <= j + 2; k++)
       if (k >= 0 && k < npe())
  astatus[k] = 1;
   }
      MPI_Scatter (astatus, 1, MPI_INT, &status, 1, MPI_INT, 0, MPI_COMM_WORLD);
      pfree (astatus,__func__,__FILE__,__LINE__);

      matrix_free (send);
      matrix_free (receive);
    }
    else {
      rcv_pid_row (sndrcv->snd, l, row);
      MPI_Gather (row, npe(), MPI_INT, NULL, npe(), MPI_INT, 0, MPI_COMM_WORLD);
      rcv_pid_row (sndrcv->rcv, l, row);
      MPI_Gather (row, npe(), MPI_INT, NULL, npe(), MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Scatter (NULL, 1, MPI_INT, &status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (status) {
      fprintf (qstderr(),
        "check_snd_rcv_matrix \"%s\" failed\n"
        "Calling debug_mpi(NULL)...\n"
        "Aborting...\n",
        name);
      fflush (qstderr());
      debug_mpi (NULL);
      MPI_Abort (MPI_COMM_WORLD, -3);
    }
  }
  pfree (row,__func__,__FILE__,__LINE__);
}

static bool has_local_child (Point point)
{ int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 767 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"

   { foreach_child()
    if (is_local(cell))
      return true; end_foreach_child(); }
  return false;
}


void mpi_boundary_update_buffers()
{ trace ("mpi_boundary_update_buffers", "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h", 776);
  if (npe() == 1)
    { ; end_trace("mpi_boundary_update_buffers", "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h", 778);  return; }

  prof_start ("mpi_boundary_update_buffers");

  MpiBoundary * m = (MpiBoundary *) mpi_boundary;
  SndRcv * mpi_level = &m->mpi_level;
  SndRcv * mpi_level_root = &m->mpi_level_root;
  SndRcv * restriction = &m->restriction;

  snd_rcv_free (mpi_level);
  snd_rcv_free (mpi_level_root);
  snd_rcv_free (restriction);

  static const unsigned short used = 1 << user;
   { foreach_cell(){

#line 792 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"
 {
    if (is_active(cell) && !is_border(cell))



      continue;

    if (cell.neighbors) {

      Array pids = {NULL, 0, 0};
      int n = locals_pids (point, &pids);
      if (n) {
  { foreach_child()
   if (is_local(cell))
     for (int i = 0, * p = (int *) pids.p; i < n; i++, p++)
       rcv_pid_append (mpi_level->snd, *p, point); end_foreach_child(); }
 pfree (pids.p,__func__,__FILE__,__LINE__);
      }

      bool locals = false;
      if (is_leaf(cell)) {
 if ((cell.pid >= 0 && cell.pid != pid())) {
   Point p = point;
    { foreach_neighbor(1)
     if ((is_local(cell) &&
   ((!is_leaf (cell) && cell.neighbors && cell.pid >= 0) || is_local_prolongation (point, p))) ||
  is_root(point))
       locals = true, foreach_neighbor_break(); end_foreach_neighbor(); }
 }
      }
      else
  { foreach_neighbor(1)
   if (is_local(cell) || is_root(point))
     locals = true, foreach_neighbor_break(); end_foreach_neighbor(); }
      if (locals)
  { foreach_child()
   if ((cell.pid >= 0 && cell.pid != pid()))
            rcv_pid_append (mpi_level->rcv, cell.pid, point),
       cell.flags |= used; end_foreach_child(); }


      if (!is_leaf(cell)) {

 if (is_local(cell)) {
   Array pids = {NULL, 0, 0};

   int n = root_pids (point, &pids);
   if (n) {
      { foreach_neighbor()
       for (int i = 0, * p = (int *) pids.p; i < n; i++, p++)
  if (cell.pid >= 0 && cell.pid != *p)
    rcv_pid_append (mpi_level_root->snd, *p, point); end_foreach_neighbor(); }

     for (int i = 0, * p = (int *) pids.p; i < n; i++, p++)
       rcv_pid_append (restriction->snd, *p, point);
     pfree (pids.p,__func__,__FILE__,__LINE__);
   }
 }

 else if ((cell.pid >= 0 && cell.pid != pid())) {
   bool root = false;
    { foreach_child()
     if (is_local(cell))
       root = true, foreach_child_break(); end_foreach_child(); }
   if (root) {
     int pid = cell.pid;
      { foreach_neighbor()
       if ((cell.pid >= 0 && cell.pid != pid()))
  rcv_pid_append (mpi_level_root->rcv, pid, point),
    cell.flags |= used; end_foreach_neighbor(); }

     rcv_pid_append (restriction->rcv, pid, point);
   }
 }
      }
    }


    if (level > 0) {
      if (is_local(cell)) {

 Array pids = {NULL, 0, 0};
 if ((aparent(0,0,0).pid >= 0 && aparent(0,0,0).pid != pid()))
   append_pid (&pids, aparent(0,0,0).pid);
 int n = root_pids (parent, &pids);
 if (n) {
   for (int i = 0, * p = (int *) pids.p; i < n; i++, p++)
     rcv_pid_append (restriction->snd, *p, point);
   pfree (pids.p,__func__,__FILE__,__LINE__);
 }
      }
      else if ((cell.pid >= 0 && cell.pid != pid())) {

 if (is_local(aparent(0,0,0)) || has_local_child (parent))
   rcv_pid_append (restriction->rcv, cell.pid, point);
      }
    }
  } } end_foreach_cell(); }





  static const unsigned short keep = 1 << (user + 1);
  for (int l = depth(); l >= 0; l--)
     { foreach_cell(){

#line 897 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"

      if (level == l) {
 if (level > 0 && (cell.pid < 0 || is_local(cell) || (cell.flags & used)))
   aparent(0,0,0).flags |= keep;
 if ((!is_leaf (cell) && cell.neighbors && cell.pid >= 0) && !(cell.flags & keep))
   coarsen_cell (point, NULL);
 cell.flags &= ~(used|keep);
 continue;
      } } end_foreach_cell(); }


  m->send->len = m->receive->len = 0;
  rcv_pid_append_pids (mpi_level->snd, m->send);
  rcv_pid_append_pids (mpi_level_root->snd, m->send);
  rcv_pid_append_pids (mpi_level->rcv, m->receive);
  rcv_pid_append_pids (mpi_level_root->rcv, m->receive);

  prof_stop();
#line 929 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"
 end_trace("mpi_boundary_update_buffers", "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h", 929); }


void mpi_boundary_refine (scalar * list)
{ trace ("mpi_boundary_refine", "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h", 933);
  prof_start ("mpi_boundary_refine");

  MpiBoundary * mpi = (MpiBoundary *) mpi_boundary;


  Array * snd = mpi->send;
  MPI_Request r[2*snd->len/sizeof(int)];
  int nr = 0;
  for (int i = 0, * dest = snd->p; i < snd->len/sizeof(int); i++,dest++) {
    int len = ((Tree *)grid)->refined.n;
    MPI_Isend (&((Tree *)grid)->refined.n, 1, MPI_INT, *dest,
        (128), MPI_COMM_WORLD, &r[nr++]);
    if (len > 0)
      MPI_Isend (((Tree *)grid)->refined.p, sizeof(Index)/sizeof(int)*len,
   MPI_INT, *dest, (128), MPI_COMM_WORLD, &r[nr++]);
  }



  Array * rcv = mpi->receive;
  Cache rerefined = {NULL, 0, 0};
  for (int i = 0, * source = rcv->p; i < rcv->len/sizeof(int); i++,source++) {
    int len;
    mpi_recv_check (&len, 1, MPI_INT, *source, (128),
      MPI_COMM_WORLD, MPI_STATUS_IGNORE,
      "mpi_boundary_refine (len)");
    if (len > 0) {
      Index p[len];
      mpi_recv_check (p, sizeof(Index)/sizeof(int)*len,
        MPI_INT, *source, (128),
        MPI_COMM_WORLD, MPI_STATUS_IGNORE,
        "mpi_boundary_refine (p)");
      Cache refined = {p, len, len};
       { foreach_cache (refined){

#line 967 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"

 if (level <= depth() && allocated(0,0,0)) {
   if (is_leaf(cell)) {
     bool neighbors = false;
      { foreach_neighbor()
       if (allocated(0,0,0) && (is_active(cell) || is_local(aparent(0,0,0))))
  neighbors = true, foreach_neighbor_break(); end_foreach_neighbor(); }

     if (neighbors)
       refine_cell (point, list, 0, &rerefined);
   }
 } } end_foreach_cache(); }
    }
  }


  if (nr)
    MPI_Waitall (nr, r, MPI_STATUSES_IGNORE);


  pfree (((Tree *)grid)->refined.p,__func__,__FILE__,__LINE__);
  ((Tree *)grid)->refined = rerefined;

  prof_stop();



  mpi_all_reduce (rerefined.n, MPI_INT, MPI_SUM);
  if (rerefined.n)
    mpi_boundary_refine (list);
 end_trace("mpi_boundary_refine", "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h", 997); }

static void check_depth()
{
#line 1032 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"
}

typedef struct {
  int refined, leaf;
} Remote;




void mpi_boundary_coarsen (int l, int too_fine)
{ trace ("mpi_boundary_coarsen", "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h", 1042);
  if (npe() == 1)
    { ; end_trace("mpi_boundary_coarsen", "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h", 1044);  return; }

  check_depth();

  assert (sizeof(Remote) == sizeof(double));

  scalar remote= new_scalar("remote");
   { foreach_cell(){

#line 1051 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"
 {
    if (level == l) {
      if (is_local(cell)) {
 ((Remote *)&val(remote,0,0,0))->refined = (!is_leaf (cell) && cell.neighbors && cell.pid >= 0);
 ((Remote *)&val(remote,0,0,0))->leaf = is_leaf(cell);
      }
      else {
 ((Remote *)&val(remote,0,0,0))->refined = true;
 ((Remote *)&val(remote,0,0,0))->leaf = false;
      }
      continue;
    }
    if (is_leaf(cell))
      continue;
  } } end_foreach_cell(); }
  mpi_boundary_level (mpi_boundary, ((scalar []){remote,{-1}}), l);

   { foreach_cell(){

#line 1068 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"
 {
    if (level == l) {
      if (!is_local(cell)) {
 if ((!is_leaf (cell) && cell.neighbors && cell.pid >= 0) && !((Remote *)&val(remote,0,0,0))->refined)
   coarsen_cell_recursive (point, NULL);
 else if (is_leaf(cell) && cell.neighbors && ((Remote *)&val(remote,0,0,0))->leaf) {
   int pid = cell.pid;
    { foreach_child()
     cell.pid = pid; end_foreach_child(); }
 }
      }
      continue;
    }
    if (is_leaf(cell))
      continue;
  } } end_foreach_cell(); }

  check_depth();

  if (l > 0) {
     { foreach_cell(){

#line 1088 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"
 {
      if (level == l) {
 val(remote,0,0,0) = is_local(cell) ? cell.neighbors : 0;
 continue;
      }
      if (is_leaf(cell))
 continue;
    } } end_foreach_cell(); }
    mpi_boundary_level (mpi_boundary, ((scalar []){remote,{-1}}), l);
     { foreach_cell(){

#line 1097 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"
 {
      if (level == l)
 if (!is_local(cell) && is_local(aparent(0,0,0)) && val(remote,0,0,0)) {
   aparent(0,0,0).flags &= ~too_fine;
   continue;
 }
      if (is_leaf(cell))
 continue;
    } } end_foreach_cell(); }
  }
 delete (((scalar []){remote,{-1}}));  end_trace("mpi_boundary_coarsen", "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h", 1107); }

static void flag_border_cells()
{
   { foreach_cell(){

#line 1111 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"
 {
    if (is_active(cell)) {
      short flags = cell.flags & ~border;
       { foreach_neighbor() {
 if (!is_local(cell) || (level > 0 && !is_local(aparent(0,0,0))))
   flags |= border, foreach_neighbor_break();

 if ((!is_leaf (cell) && cell.neighbors && cell.pid >= 0))
    { foreach_child()
     if (!is_local(cell))
       flags |= border, foreach_child_break(); end_foreach_child(); }
 if (flags & border)
   foreach_neighbor_break();
      } end_foreach_neighbor(); }
      cell.flags = flags;
    }
    else {
      cell.flags &= ~border;

    }
    if (is_leaf(cell)) {
      if (cell.neighbors) {
  { foreach_child()
   cell.flags &= ~border; end_foreach_child(); }
 if (is_border(cell)) {
   bool remote = false;
    { foreach_neighbor (2/2)
     if (!is_local(cell))
       remote = true, foreach_neighbor_break(); end_foreach_neighbor(); }
   if (remote)
      { foreach_child()
       cell.flags |= border; end_foreach_child(); }
 }
      }
      continue;
    }
  } } end_foreach_cell(); }
}

static int balanced_pid (long index, long nt, int nproc)
{
  long ne = max(1, nt/nproc), nr = nt % nproc;
  int pid = index < nr*(ne + 1) ?
    index/(ne + 1) :
    nr + (index - nr*(ne + 1))/ne;
  return min(nproc - 1, pid);
}



void mpi_partitioning()
{ trace ("mpi_partitioning", "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h", 1162);
  prof_start ("mpi_partitioning");

  long nt = 0;
   { foreach(){

#line 1166 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"

    nt++; } end_foreach(); }


  long i = 0;
  ((Tree *)grid)->dirty = true;
   { foreach_cell_post (is_active (cell)){

#line 1172 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"

    if (is_active (cell)) {
      if (is_leaf (cell)) {
 cell.pid = balanced_pid (i++, nt, npe());
 if (cell.neighbors > 0) {
   int pid = cell.pid;
    { foreach_child()
     cell.pid = pid; end_foreach_child(); }
 }
 if (!is_local(cell))
   cell.flags &= ~active;
      }
      else {
 cell.pid = child(0,0,0).pid;
 bool inactive = true;
  { foreach_child()
   if (is_active(cell))
     inactive = false, foreach_child_break(); end_foreach_child(); }
 if (inactive)
   cell.flags &= ~active;
      }
    } } end_foreach_cell_post(); }

  flag_border_cells();

  prof_stop();

  mpi_boundary_update_buffers();
 end_trace("mpi_partitioning", "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h", 1200); }

void restore_mpi (FILE * fp, scalar * list1)
{
  long index = 0, nt = 0, start = ftell (fp);
  scalar size= new_scalar("size"), * list = list_concat (((scalar []){size,{-1}}), list1);;
  long offset = sizeof(double)*list_len(list);


  static const unsigned short set = 1 << user;
  scalar * listm = is_constant(cm) ? NULL : (scalar *)((vector []){{fm.x,fm.y,fm.z},{{-1},{-1},{-1}}});
   { foreach_cell(){

#line 1211 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"

    if (balanced_pid (index, nt, npe()) <= pid()) {
      unsigned flags;
      if (fread (&flags, sizeof(unsigned), 1, fp) != 1) {
 fprintf (qstderr(), "restore(): error: expecting 'flags'\n");
 exit (1);
      }
      if (list) for (scalar s = *list, *_i69 = list; ((scalar *)&s)->i >= 0; s = *++_i69) {
 double val;
 if (fread (&val, sizeof(double), 1, fp) != 1) {
   fprintf (qstderr(), "restore(): error: expecting scalar\n");
   exit (1);
 }
 if (s.i != INT_MAX)
   val(s,0,0,0) = val;
      }
      if (level == 0)
 nt = val(size,0,0,0);
      cell.pid = balanced_pid (index, nt, npe());
      cell.flags |= set;
      if (!(flags & leaf) && is_leaf(cell)) {
 if (balanced_pid (index + val(size,0,0,0) - 1, nt, npe()) < pid()) {
   fseek (fp, (sizeof(unsigned) + offset)*(val(size,0,0,0) - 1), SEEK_CUR);
   index += val(size,0,0,0);
   continue;
 }
 refine_cell (point, listm, 0, NULL);
      }
      index++;
      if (is_leaf(cell))
 continue;
    } } end_foreach_cell(); }


  fseek (fp, start, SEEK_SET);
  index = 0;
   { foreach_cell(){

#line 1247 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"
 {
    unsigned flags;
    if (fread (&flags, sizeof(unsigned), 1, fp) != 1) {
      fprintf (qstderr(), "restore(): error: expecting 'flags'\n");
      exit (1);
    }
    if (cell.flags & set)
      fseek (fp, offset, SEEK_CUR);
    else {
      if (list) for (scalar s = *list, *_i70 = list; ((scalar *)&s)->i >= 0; s = *++_i70) {
 double val;
 if (fread (&val, sizeof(double), 1, fp) != 1) {
   fprintf (qstderr(), "restore(): error: expecting a scalar\n");
   exit (1);
 }
 if (s.i != INT_MAX)
   val(s,0,0,0) = val;
      }
      cell.pid = balanced_pid (index, nt, npe());
      if (is_leaf(cell) && cell.neighbors) {
 int pid = cell.pid;
  { foreach_child()
   cell.pid = pid; end_foreach_child(); }
      }
    }
    if (!(flags & leaf) && is_leaf(cell)) {
      bool locals = false;
       { foreach_neighbor(1)
 if ((cell.flags & set) && (is_local(cell) || is_root(point)))
   locals = true, foreach_neighbor_break(); end_foreach_neighbor(); }
      if (locals)
 refine_cell (point, listm, 0, NULL);
      else {
 fseek (fp, (sizeof(unsigned) + offset)*(val(size,0,0,0) - 1), SEEK_CUR);
 index += val(size,0,0,0);
 continue;
      }
    }
    index++;
    if (is_leaf(cell))
      continue;
  } } end_foreach_cell(); }


   { foreach_cell_post (is_active (cell)){

#line 1291 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"
 {
    cell.flags &= ~set;
    if (is_active (cell)) {
      if (is_leaf (cell)) {
 if (cell.neighbors > 0) {
   int pid = cell.pid;
    { foreach_child()
     cell.pid = pid; end_foreach_child(); }
 }
 if (!is_local(cell))
   cell.flags &= ~active;
      }
      else if (!is_local(cell)) {
 bool inactive = true;
  { foreach_child()
   if (is_active(cell))
     inactive = false, foreach_child_break(); end_foreach_child(); }
 if (inactive)
   cell.flags &= ~active;
      }
    }
  } } end_foreach_cell_post(); }

  flag_border_cells();

  mpi_boundary_update (list);
  pfree (list,__func__,__FILE__,__LINE__);
 delete (((scalar []){size,{-1}})); }
#line 1340 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"

double z_indexing (scalar index, bool leaves)
{ trace ("z_indexing", "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h", 1342);



  scalar size= new_scalar("size");
  subtree_size (size, leaves);






  double maxi = -1.;
  if (pid() == 0)
     { foreach_level(0){

#line 1356 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"

      maxi = val(size,0,0,0) - 1.; } end_foreach_level(); }




   { foreach_level(0){

#line 1362 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"

    val(index,0,0,0) = 0; } end_foreach_level(); }
  for (int l = 0; l < depth(); l++) {
    { Boundary ** _i = boundaries, * _b; while ((_b = *_i++)) if (_b->restriction) _b->restriction (_b, ((scalar []){index,{-1}}), l); };
     { foreach_cell(){

#line 1366 "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h"
 {
      if (level == l) {
 if (is_leaf(cell)) {
   if (is_local(cell) && cell.neighbors) {
     int i = val(index,0,0,0);
      { foreach_child()
       val(index,0,0,0) = i; end_foreach_child(); }
   }
 }
 else {
   bool loc = is_local(cell);
   if (!loc)
      { foreach_child()
       if (is_local(cell))
  loc = true, foreach_child_break(); end_foreach_child(); }
   if (loc) {
     int i = val(index,0,0,0) + !leaves;
      { foreach_child() {
       val(index,0,0,0) = i;
       i += val(size,0,0,0);
     } end_foreach_child(); }
   }
 }
 continue;
      }
      if (is_leaf(cell))
 continue;
    } } end_foreach_cell(); }
  }
  { Boundary ** _i = boundaries, * _b; while ((_b = *_i++)) if (_b->restriction) _b->restriction (_b, ((scalar []){index,{-1}}), depth()); };

  { double _ret =  maxi; delete (((scalar []){size,{-1}}));  end_trace("z_indexing", "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h", 1397);  return _ret; }
 delete (((scalar []){size,{-1}}));  end_trace("z_indexing", "/home/miro3/Documents/Programming/basilisk/src/grid/tree-mpi.h", 1398); }
#line 1744 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
#line 1 "grid/balance.h"
#line 1 "/home/miro3/Documents/Programming/basilisk/src/grid/balance.h"


typedef struct {
  short leaf, prolongation;
  int pid;
} NewPid;



#if TRASH
# define is_newpid() (!isnan(val(newpid,0,0,0)) && ((NewPid *)&val(newpid,0,0,0))->pid > 0)
#else
# define is_newpid() (((NewPid *)&val(newpid,0,0,0))->pid > 0)
#endif

Array * linear_tree (size_t size, scalar newpid)
{
  const unsigned short sent = 1 << user, next = 1 << (user + 1);
  Array * a = array_new();

   { foreach_cell_post_all (true){

#line 21 "/home/miro3/Documents/Programming/basilisk/src/grid/balance.h"

    if (level > 0 && (cell.flags & (sent|next)))
      aparent(0,0,0).flags |= next; } end_foreach_cell_post_all(); }

  bool empty = true;
   { foreach_cell_all(){

#line 26 "/home/miro3/Documents/Programming/basilisk/src/grid/balance.h"
 {
    if (cell.flags & sent) {
      array_append (a, &cell, size);
      cell.flags &= ~sent;
      empty = false;
    }
    else {
      if (cell.pid >= 0 && ((NewPid *)&val(newpid,0,0,0))->leaf)
 assert (is_leaf(cell));
      if ((!is_leaf (cell) && cell.neighbors && cell.pid >= 0)) {


 bool prolo = false;
  { foreach_child()
   if (((NewPid *)&val(newpid,0,0,0))->prolongation)
     prolo = true; end_foreach_child(); }
 if (prolo) {

   cell.flags |= leaf;
   array_append (a, &cell, sizeof(Cell));
   cell.flags &= ~leaf;
 }
 else
   array_append (a, &cell, sizeof(Cell));
      }
      else
 array_append (a, &cell, sizeof(Cell));
    }
    if (cell.flags & next)
      cell.flags &= ~next;
    else
      continue;
  } } end_foreach_cell_all(); }

  if (empty)
    a->len = 0;
  return a;
}

#define foreach_tree(t, size, list)\
{\
  const unsigned short _sent = 1 << user, _next = 1 << (user + 1);\
  scalar * _list = list;\
  char * _i = (char *) (t)->p;\
  foreach_cell_all() {\
    Cell * c = (Cell *) _i;\
    if (c->flags & _sent) {\
      _i += size;\

#line 74


#define end_foreach_tree()\
    }\
    else\
      _i += sizeof(Cell);\
    if (c->flags & _next) {\
      assert (c->neighbors);\
      if (!(c->flags & leaf) && is_leaf(cell) &&\
   (!is_newpid() || !((NewPid *)&val(newpid,0,0,0))->leaf))\
\
 refine_cell (point, _list, 0, NULL);\
      else if (!cell.neighbors)\
\
 alloc_children (point);\
    }\
    else\
      continue;\
  } end_foreach_cell_all();\
}\

#line 94


Array * neighborhood (scalar newpid, int nextpid, FILE * fp)
{
  const unsigned short sent = 1 << user;
   { foreach_cell(){

#line 99 "/home/miro3/Documents/Programming/basilisk/src/grid/balance.h"
 {

    bool root = false;
    if ((!is_local(cell) || ((NewPid *)&val(newpid,0,0,0))->pid - 1 != nextpid) && (!is_leaf (cell) && cell.neighbors && cell.pid >= 0)) {
       { foreach_child()
 if (is_local(cell) && ((NewPid *)&val(newpid,0,0,0))->pid - 1 == nextpid)
   root = true, foreach_child_break(); end_foreach_child(); }
      if (root && cell.pid != nextpid) {
  { foreach_neighbor()
   if (cell.pid != nextpid && is_newpid()) {
     if (fp)
       fprintf (fp, "%g %g %g %d %d root\n",
         x, y, z, ((NewPid *)&val(newpid,0,0,0))->pid - 1, cell.pid);
     cell.flags |= sent;
   } end_foreach_neighbor(); }
      }
    }

    if ((is_local(cell) && ((NewPid *)&val(newpid,0,0,0))->pid - 1 == nextpid) || root) {
       { foreach_neighbor(1)
 if (cell.neighbors && cell.pid != nextpid)
    { foreach_child()
     if (cell.pid != nextpid && is_newpid()) {
       if (fp)
  fprintf (fp, "%g %g %g %d %d nextpid\n",
    x, y, z, ((NewPid *)&val(newpid,0,0,0))->pid - 1, cell.pid);
       cell.flags |= sent;
     } end_foreach_child(); } end_foreach_neighbor(); }
    }
    if (is_leaf(cell))
      continue;
  } } end_foreach_cell(); }

  return linear_tree (sizeof(Cell) + datasize, newpid);
}

static void send_tree (Array * a, int to, MPI_Request * r)
{
  MPI_Isend (&a->len, 1, MPI_LONG, to, (256), MPI_COMM_WORLD, &r[0]);
  if (a->len > 0) {
    MPI_Isend (a->p, a->len, MPI_BYTE, to, (256), MPI_COMM_WORLD, &r[1]);
    ((Tree *)grid)->dirty = true;
  }
}

static void receive_tree (int from, scalar newpid, FILE * fp)
{
  Array a;
  mpi_recv_check (&a.len, 1, MPI_LONG, from, (256),
    MPI_COMM_WORLD, MPI_STATUS_IGNORE, "receive_tree (len)");
  if (a.len > 0) {
    a.p = pmalloc (a.len,__func__,__FILE__,__LINE__);
    if (fp)
      fprintf (fp, "receiving %ld from %d\n", a.len, from);
    mpi_recv_check (a.p, a.len, MPI_BYTE, from, (256),
      MPI_COMM_WORLD, MPI_STATUS_IGNORE, "receive_tree (p)");

     { foreach_tree (&a, sizeof(Cell) + datasize, NULL){

#line 156 "/home/miro3/Documents/Programming/basilisk/src/grid/balance.h"
 {
      memcpy (((char *)&cell) + sizeof(Cell), ((char *)c) + sizeof(Cell),
       datasize);
      assert (((NewPid *)&val(newpid,0,0,0))->pid > 0);
      if (fp)
 fprintf (fp, "%g %g %g %d %d %d %d %d %d recv\n",
   x, y, z, ((NewPid *)&val(newpid,0,0,0))->pid - 1, cell.pid,
   c->flags & leaf,
   cell.flags & leaf, from, ((NewPid *)&val(newpid,0,0,0))->leaf);
    } } end_foreach_tree(); }
    pfree (a.p,__func__,__FILE__,__LINE__);
    ((Tree *)grid)->dirty = true;
  }
}

static void wait_tree (Array * a, MPI_Request * r)
{
  MPI_Wait (&r[0], MPI_STATUS_IGNORE);
  if (a->len > 0)
    MPI_Wait (&r[1], MPI_STATUS_IGNORE);
}

static void check_flags()
{







}

struct {
  int min;
  bool leaves;

  int npe;
} mpi = {
  1,
  true
};


bool balance()
{ trace ("balance", "/home/miro3/Documents/Programming/basilisk/src/grid/balance.h", 201);
  if (npe() == 1)
    { bool _ret =  false; end_trace("balance", "/home/miro3/Documents/Programming/basilisk/src/grid/balance.h", 203);  return _ret; }

  assert (sizeof(NewPid) == sizeof(double));

  check_flags();

  long nl = 0, nt = 0;
   { foreach_cell(){

#line 210 "/home/miro3/Documents/Programming/basilisk/src/grid/balance.h"
 {
    if (is_local(cell)) {
      nt++;
      if (is_leaf(cell))
 nl++;
    }
    if (is_leaf(cell))
      continue;
  } } end_foreach_cell(); }

  grid->n = grid->tn = nl;
  grid->maxdepth = depth();
  long nmin = nl, nmax = nl;

  mpi_all_reduce (nmax, MPI_LONG, MPI_MAX);
  mpi_all_reduce (nmin, MPI_LONG, MPI_MIN);
  mpi_all_reduce (grid->tn, MPI_LONG, MPI_SUM);
  mpi_all_reduce (grid->maxdepth, MPI_INT, MPI_MAX);
  if (mpi.leaves)
    nt = grid->tn;
  else
    mpi_all_reduce (nt, MPI_LONG, MPI_SUM);

  long ne = max(1, nt/npe());

  if (ne < mpi.min) {
    mpi.npe = max(1, nt/mpi.min);
    ne = max(1, nt/mpi.npe);
  }
  else
    mpi.npe = npe();

  if (nmax - nmin <= 1)
    { bool _ret =  false; end_trace("balance", "/home/miro3/Documents/Programming/basilisk/src/grid/balance.h", 243);  return _ret; }

  scalar newpid= new_scalar("newpid");
  double zn = z_indexing (newpid, mpi.leaves);
  if (pid() == 0)
    assert (zn + 1 == nt);

  FILE * fp = NULL;
#line 260 "/home/miro3/Documents/Programming/basilisk/src/grid/balance.h"
  bool next = false, prev = false;
   { foreach_cell_all(){

#line 261 "/home/miro3/Documents/Programming/basilisk/src/grid/balance.h"
 {
    if (is_local(cell)) {
      int pid = balanced_pid (val(newpid,0,0,0), nt, mpi.npe);
      pid = clamp (pid, cell.pid - 1, cell.pid + 1);
      if (pid == pid() + 1)
 next = true;
      else if (pid == pid() - 1)
 prev = true;
      ((NewPid *)&val(newpid,0,0,0))->pid = pid + 1;
      ((NewPid *)&val(newpid,0,0,0))->leaf = is_leaf(cell);
      ((NewPid *)&val(newpid,0,0,0))->prolongation = (!is_leaf(cell) && !cell.neighbors && cell.pid >= 0);
      if (fp)
 fprintf (fp, "%g %g %d %d newpid\n", x, y, ((NewPid *)&val(newpid,0,0,0))->pid - 1, cell.pid);
    }
    else
      val(newpid,0,0,0) = 0;
  } } end_foreach_cell_all(); }
  for (int l = 0; l <= depth(); l++)
    { Boundary ** _i = boundaries, * _b; while ((_b = *_i++)) if (_b->level) _b->level (_b, ((scalar []){newpid,{-1}}), l); };
#line 304 "/home/miro3/Documents/Programming/basilisk/src/grid/balance.h"
  Array * anext = next ? neighborhood (newpid, pid() + 1, fp) : array_new();
  Array * aprev = prev ? neighborhood (newpid, pid() - 1, fp) : array_new();

  if (fp)
    fflush (fp);

  check_flags();


  MPI_Request rprev[2], rnext[2];
  if (pid() > 0)
    send_tree (aprev, pid() - 1, rprev);
  if (pid() < npe() - 1)
    send_tree (anext, pid() + 1, rnext);


  if (pid() < npe() - 1)
    receive_tree (pid() + 1, newpid, fp);
  if (pid() > 0)
    receive_tree (pid() - 1, newpid, fp);


  if (pid() > 0)
    wait_tree (aprev, rprev);
  array_free (aprev);
  if (pid() < npe() - 1)
    wait_tree (anext, rnext);
  array_free (anext);

  if (fp)
    fflush (fp);


  int pid_changed = false;
   { foreach_cell_all(){

#line 338 "/home/miro3/Documents/Programming/basilisk/src/grid/balance.h"
 {
    if (cell.pid >= 0) {
      if (is_newpid()) {
 if (fp)
   fprintf (fp, "%g %g %g %d %d %d %d %d new\n",
     x, y, z, ((NewPid *)&val(newpid,0,0,0))->pid - 1, cell.pid,
     is_leaf(cell), cell.neighbors, ((NewPid *)&val(newpid,0,0,0))->leaf);
 if (cell.pid != ((NewPid *)&val(newpid,0,0,0))->pid - 1) {
   cell.pid = ((NewPid *)&val(newpid,0,0,0))->pid - 1;
   cell.flags &= ~(active|border);
   if (is_local(cell))
     cell.flags |= active;
   pid_changed = true;
 }
 if (((NewPid *)&val(newpid,0,0,0))->leaf && !is_leaf(cell) && cell.neighbors)
   coarsen_cell_recursive (point, NULL);
      }
      else if (level > 0 && ((NewPid *)&coarse(newpid,0,0,0))->leaf)
 cell.pid = aparent(0,0,0).pid;
    }

    if (!cell.neighbors && allocated_child(0,0,0)) {
      if (fp)
 fprintf (fp, "%g %g %g %d %d freechildren\n",
   x, y, z, ((NewPid *)&val(newpid,0,0,0))->pid - 1, cell.pid);
      free_children (point);
    }
  } } end_foreach_cell_all(); }

  if (((Tree *)grid)->dirty || pid_changed) {


     { foreach_cell_post (!is_leaf (cell)){

#line 370 "/home/miro3/Documents/Programming/basilisk/src/grid/balance.h"

      if (!is_leaf(cell) && !is_local(cell)) {
 unsigned short flags = cell.flags & ~active;
  { foreach_child()
   if (is_active(cell))
     flags |= active, foreach_child_break(); end_foreach_child(); }
 cell.flags = flags;
      } } end_foreach_cell_post(); }

    flag_border_cells();
    pid_changed = true;
  }

  if (fp)
    fclose (fp);

  mpi_all_reduce (pid_changed, MPI_INT, MPI_MAX);
  if (pid_changed)
    mpi_boundary_update_buffers();

  { bool _ret =  pid_changed; delete (((scalar []){newpid,{-1}}));  end_trace("balance", "/home/miro3/Documents/Programming/basilisk/src/grid/balance.h", 390);  return _ret; }
 delete (((scalar []){newpid,{-1}}));  end_trace("balance", "/home/miro3/Documents/Programming/basilisk/src/grid/balance.h", 391); }

void mpi_boundary_update (scalar * list)
{
  mpi_boundary_update_buffers();
  grid->tn = 0;
  boundary (list);
  while (balance());
}
#line 1745 "/home/miro3/Documents/Programming/basilisk/src/grid/tree.h"
#else
void mpi_boundary_refine (scalar * list){}
void mpi_boundary_coarsen (int a, int b){}
void mpi_boundary_update (scalar * list) {
  boundary (list);
}
#endif
#line 4 "/home/miro3/Documents/Programming/basilisk/src/grid/octree.h"

void octree_methods() {
  tree_methods();
}
#line 2 "probing.c"
#line 1 "vtk.h"
#line 1 "/home/miro3/Documents/Programming/basilisk/src/vtk.h"
void output_vtk (scalar * list, int n, FILE * fp, bool linear)
{
  fputs ("# vtk DataFile Version 2.0\n"
  "Basilisk\n"
  "ASCII\n"
  "DATASET STRUCTURED_GRID\n", fp);
  fprintf (fp, "DIMENSIONS %d %d 1\n", n, n);
  fprintf (fp, "POINTS %d double\n", n*n);

  double fn = n;
  double Delta = L0/fn;
  for (int i = 0; i < n; i++) {
    double x = Delta*i + X0 + Delta/2.;
    for (int j = 0; j < n; j++) {
      double y = Delta*j + Y0 + Delta/2.;
      fprintf (fp, "%g %g 0\n", x, y);
    }
  }
  fprintf (fp, "POINT_DATA %d\n", n*n);
  if (list) for (scalar s = *list, *_i71 = list; ((scalar *)&s)->i >= 0; s = *++_i71) {
    fprintf (fp, "SCALARS %s double\n", _attribute[s.i].name);
    fputs ("LOOKUP_TABLE default\n", fp);
    double fn = n;
    double Delta = L0/fn;
    for (int i = 0; i < n; i++) {
      double x = Delta*i + X0 + Delta/2.;
      for (int j = 0; j < n; j++) {
 double y = Delta*j + Y0 + Delta/2., v;
 if (linear)
   v = interpolate ((struct _interpolate){s, x, y});
 else {
   Point point = locate ((struct _locate){x, y});  int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 32 "/home/miro3/Documents/Programming/basilisk/src/vtk.h"

   v = point.level >= 0 ? val(s,0,0,0) : nodata;
 }
 fprintf (fp, "%g\n", v);
      }
    }
  }
  fflush (fp);
}
#line 57 "/home/miro3/Documents/Programming/basilisk/src/vtk.h"
void output_pvtu_ascii (scalar * list, vector * vlist, int n, FILE * fp, char * subname)
{
    fputs ("<?xml version=\"1.0\"?>\n"
    "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n", fp);
    fputs ("\t <PUnstructuredGrid GhostLevel=\"0\">\n", fp);
    fputs ("\t\t\t <PCellData Scalars=\"scalars\">\n", fp);
    if (list) for (scalar s = *list, *_i72 = list; ((scalar *)&s)->i >= 0; s = *++_i72) {
      fprintf (fp,"\t\t\t\t <PDataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n", _attribute[s.i].name);
      fputs ("\t\t\t\t </PDataArray>\n", fp);
    }
    if (vlist) for (vector v = *vlist, *_i73 = vlist; ((scalar *)&v)->i >= 0; v = *++_i73) {
      fprintf (fp,"\t\t\t\t <PDataArray type=\"Float64\" NumberOfComponents=\"3\" Name=\"%s\" format=\"ascii\">\n", _attribute[v.x.i].name);
      fputs ("\t\t\t\t </PDataArray>\n", fp);
    }
    fputs ("\t\t\t </PCellData>\n", fp);
    fputs ("\t\t\t <PPoints>\n", fp);
    fputs ("\t\t\t\t <PDataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n", fp);
    fputs ("\t\t\t\t </PDataArray>\n", fp);
    fputs ("\t\t\t </PPoints>\n", fp);

    for (int i = 0; i < npe(); i++)
      fprintf (fp, "<Piece Source=\"%s_n%3.3d.vtu\"/> \n", subname, i);

    fputs ("\t </PUnstructuredGrid>\n", fp);
    fputs ("</VTKFile>\n", fp);
}
#line 92 "/home/miro3/Documents/Programming/basilisk/src/vtk.h"
void output_vtu_ascii_foreach (scalar * list, vector * vlist, int n, FILE * fp, bool linear)
{





  scalar marker= new_vertex_scalar("marker");
  int no_points = 0, no_cells=0 ;
   { foreach_vertex(){

#line 101 "/home/miro3/Documents/Programming/basilisk/src/vtk.h"
{
    val(marker,0,0,0) = _k;
    no_points += 1;
  } } end_foreach_vertex(); }
   { foreach(){

#line 105 "/home/miro3/Documents/Programming/basilisk/src/vtk.h"
{
    no_cells += 1;
  } } end_foreach(); }

  fputs ("<?xml version=\"1.0\"?>\n"
  "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n", fp);
  fputs ("\t <UnstructuredGrid>\n", fp);
  fprintf (fp,"\t\t <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n", no_points, no_cells);
  fputs ("\t\t\t <CellData Scalars=\"scalars\">\n", fp);
  if (list) for (scalar s = *list, *_i74 = list; ((scalar *)&s)->i >= 0; s = *++_i74) {
    fprintf (fp,"\t\t\t\t <DataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n", _attribute[s.i].name);
     { foreach(){

#line 116 "/home/miro3/Documents/Programming/basilisk/src/vtk.h"
{
      fprintf (fp, "%g\n", val(s,0,0,0));
    } } end_foreach(); }
    fputs ("\t\t\t\t </DataArray>\n", fp);
  }
  if (vlist) for (vector v = *vlist, *_i75 = vlist; ((scalar *)&v)->i >= 0; v = *++_i75) {
    fprintf (fp,"\t\t\t\t <DataArray type=\"Float64\" NumberOfComponents=\"3\" Name=\"%s\" format=\"ascii\">\n", _attribute[v.x.i].name);
     { foreach(){

#line 123 "/home/miro3/Documents/Programming/basilisk/src/vtk.h"
{




      fprintf (fp, "%g %g %g\n", val(v.x,0,0,0), val(v.y,0,0,0), val(v.z,0,0,0));

    } } end_foreach(); }
    fputs ("\t\t\t\t </DataArray>\n", fp);
  }
  fputs ("\t\t\t </CellData>\n", fp);
  fputs ("\t\t\t <Points>\n", fp);
  fputs ("\t\t\t\t <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n", fp);
   { foreach_vertex(){

#line 136 "/home/miro3/Documents/Programming/basilisk/src/vtk.h"
{




    fprintf (fp, "%g %g %g\n", x, y, z);

  } } end_foreach_vertex(); }
  fputs ("\t\t\t\t </DataArray>\n", fp);
  fputs ("\t\t\t </Points>\n", fp);
  fputs ("\t\t\t <Cells>\n", fp);
  fputs ("\t\t\t\t <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n", fp);
   { foreach(){

#line 148 "/home/miro3/Documents/Programming/basilisk/src/vtk.h"
{
#line 163 "/home/miro3/Documents/Programming/basilisk/src/vtk.h"
    int ape1 = val(marker,0,0,0);
    int ape2 = val(marker,1,0,0);
    int ape3 = val(marker,1,1,0);
    int ape4 = val(marker,0,1,0);
    int ape5 = val(marker,0,0,1);
    int ape6 = val(marker,1,0,1);
    int ape7 = val(marker,1,1,1);
    int ape8 = val(marker,0,1,1);
    fprintf (fp, "%u %u %u %u %u %u %u %u\n", ape1, ape2, ape3, ape4, ape5, ape6, ape7, ape8);


  } } end_foreach(); }
  fputs ("\t\t\t\t </DataArray>\n", fp);
  fputs ("\t\t\t\t <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n", fp);

  for (int i = 1; i < no_cells+1; i++){




    fprintf (fp, "%d \n", i*8);

  }
  fputs ("\t\t\t\t </DataArray>\n", fp);
  fputs ("\t\t\t\t <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n", fp);
   { foreach(){

#line 188 "/home/miro3/Documents/Programming/basilisk/src/vtk.h"
{




    fputs ("12 \n", fp);

  } } end_foreach(); }
  fputs ("\t\t\t\t </DataArray>\n", fp);
  fputs ("\t\t\t </Cells>\n", fp);
  fputs ("\t\t </Piece>\n", fp);
  fputs ("\t </UnstructuredGrid>\n", fp);
  fputs ("</VTKFile>\n", fp);
  fflush (fp);



 delete (((scalar []){marker,{-1}})); }






void output_pvtu_bin (scalar * list, vector * vlist, int n, FILE * fp, char * subname)
{
    fputs ("<?xml version=\"1.0\"?>\n"
    "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n", fp);
    fputs ("\t <PUnstructuredGrid GhostLevel=\"0\">\n", fp);
    fputs ("\t\t\t <PCellData Scalars=\"scalars\">\n", fp);
    if (list) for (scalar s = *list, *_i76 = list; ((scalar *)&s)->i >= 0; s = *++_i76) {
      fprintf (fp,"\t\t\t\t <PDataArray type=\"Float64\" Name=\"%s\" format=\"appended\">\n", _attribute[s.i].name);
      fputs ("\t\t\t\t </PDataArray>\n", fp);
    }
    if (vlist) for (vector v = *vlist, *_i77 = vlist; ((scalar *)&v)->i >= 0; v = *++_i77) {
      fprintf (fp,"\t\t\t\t <PDataArray type=\"Float64\" NumberOfComponents=\"3\" Name=\"%s\" format=\"appended\">\n", _attribute[v.x.i].name);
      fputs ("\t\t\t\t </PDataArray>\n", fp);
    }
    fputs ("\t\t\t </PCellData>\n", fp);
    fputs ("\t\t\t <PPoints>\n", fp);
    fputs ("\t\t\t\t <PDataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n", fp);
    fputs ("\t\t\t\t </PDataArray>\n", fp);
    fputs ("\t\t\t </PPoints>\n", fp);

    for (int i = 0; i < npe(); i++)
      fprintf (fp, "<Piece Source=\"%s_n%3.3d.vtu\"/> \n", subname, i);

    fputs ("\t </PUnstructuredGrid>\n", fp);
    fputs ("</VTKFile>\n", fp);
}
#line 247 "/home/miro3/Documents/Programming/basilisk/src/vtk.h"
void output_vtu_bin_foreach (scalar * list, vector * vlist, int n, FILE * fp, bool linear)
{





  scalar marker= new_vertex_scalar("marker");
  int no_points = 0, no_cells=0 ;
   { foreach_vertex(){

#line 256 "/home/miro3/Documents/Programming/basilisk/src/vtk.h"
{
    val(marker,0,0,0) = _k;
    no_points += 1;
  } } end_foreach_vertex(); }
   { foreach(){

#line 260 "/home/miro3/Documents/Programming/basilisk/src/vtk.h"
{
    no_cells += 1;
  } } end_foreach(); }

  fputs ("<?xml version=\"1.0\"?>\n"
  "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n", fp);
  fputs ("\t <UnstructuredGrid>\n", fp);
  fprintf (fp,"\t\t <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n", no_points, no_cells);
  fputs ("\t\t\t <CellData Scalars=\"scalars\">\n", fp);

  int count = 0;
  if (list) for (scalar s = *list, *_i78 = list; ((scalar *)&s)->i >= 0; s = *++_i78) {
    fprintf (fp,"\t\t\t\t <DataArray type=\"Float64\" Name=\"%s\" format=\"appended\" offset=\"%d\">\n", _attribute[s.i].name,count);
    count += ((no_cells)+1)*8;
    fputs ("\t\t\t\t </DataArray>\n", fp);
  }
  if (vlist) for (vector v = *vlist, *_i79 = vlist; ((scalar *)&v)->i >= 0; v = *++_i79) {
    fprintf (fp,"\t\t\t\t <DataArray type=\"Float64\" Name=\"%s\" NumberOfComponents=\"3\"  format=\"appended\" offset=\"%d\">\n", _attribute[v.x.i].name,count);
    count += ((no_cells*3)+1)*8;
    fputs ("\t\t\t\t </DataArray>\n", fp);
  }
  fputs ("\t\t\t </CellData>\n", fp);
  fputs ("\t\t\t <Points>\n", fp);
  fprintf (fp,"\t\t\t\t <DataArray type=\"Float64\" NumberOfComponents=\"3\"  format=\"appended\" offset=\"%d\">\n",count);
  count += ((no_points*3)+1)*8;
  fputs ("\t\t\t\t </DataArray>\n", fp);
  fputs ("\t\t\t </Points>\n", fp);
  fputs ("\t\t\t <Cells>\n", fp);
  fputs ("\t\t\t\t <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n", fp);
   { foreach(){

#line 289 "/home/miro3/Documents/Programming/basilisk/src/vtk.h"
{
#line 304 "/home/miro3/Documents/Programming/basilisk/src/vtk.h"
    int ape1 = val(marker,0,0,0);
    int ape2 = val(marker,1,0,0);
    int ape3 = val(marker,1,1,0);
    int ape4 = val(marker,0,1,0);
    int ape5 = val(marker,0,0,1);
    int ape6 = val(marker,1,0,1);
    int ape7 = val(marker,1,1,1);
    int ape8 = val(marker,0,1,1);
    fprintf (fp, "%u %u %u %u %u %u %u %u\n", ape1, ape2, ape3, ape4, ape5, ape6, ape7, ape8);


  } } end_foreach(); }
  fputs ("\t\t\t\t </DataArray>\n", fp);
  fputs ("\t\t\t\t <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n", fp);
  for (int i = 1; i < no_cells+1; i++){




    fprintf (fp, "%d \n", i*8);

  }
  fputs ("\t\t\t\t </DataArray>\n", fp);
  fputs ("\t\t\t\t <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n", fp);
   { foreach(){

#line 328 "/home/miro3/Documents/Programming/basilisk/src/vtk.h"
{




    fputs ("12 \n", fp);

  } } end_foreach(); }
  fputs ("\t\t\t\t </DataArray>\n", fp);
  fputs ("\t\t\t </Cells>\n", fp);
  fputs ("\t\t </Piece>\n", fp);
  fputs ("\t </UnstructuredGrid>\n", fp);
  fputs ("\t <AppendedData encoding=\"raw\">\n", fp);
  fputs ("_", fp);
  unsigned long long block_len=no_cells*8;



  if (list) for (scalar s = *list, *_i80 = list; ((scalar *)&s)->i >= 0; s = *++_i80) {
    fwrite (&block_len, sizeof (unsigned long long), 1, fp);
     { foreach(){

#line 348 "/home/miro3/Documents/Programming/basilisk/src/vtk.h"

      fwrite (&val(s,0,0,0), sizeof (double), 1, fp); } end_foreach(); }
  }
  block_len=no_cells*8*3;
  if (vlist) for (vector v = *vlist, *_i81 = vlist; ((scalar *)&v)->i >= 0; v = *++_i81) {
    fwrite (&block_len, sizeof (unsigned long long), 1, fp);
     { foreach(){

#line 354 "/home/miro3/Documents/Programming/basilisk/src/vtk.h"
{
      fwrite (&val(v.x,0,0,0), sizeof (double), 1, fp);
      fwrite (&val(v.y,0,0,0), sizeof (double), 1, fp);




      fwrite (&val(v.z,0,0,0), sizeof (double), 1, fp);

    } } end_foreach(); }
  }
  block_len=no_points*8*3;
  fwrite (&block_len, sizeof (unsigned long long), 1, fp);
   { foreach_vertex(){

#line 367 "/home/miro3/Documents/Programming/basilisk/src/vtk.h"
{
    fwrite (&x, sizeof (double), 1, fp);
    fwrite (&y, sizeof (double), 1, fp);
    fwrite (&z, sizeof (double), 1, fp);
  } } end_foreach_vertex(); }
  fputs ("\t\n", fp);
  fputs ("\t </AppendedData>\n", fp);
  fputs ("</VTKFile>\n", fp);
  fflush (fp);



 delete (((scalar []){marker,{-1}})); }
#line 3 "probing.c"
#include <math.h>


typedef struct {

  double x;
  double y;
  double z;
} GPoint;


typedef struct {


  GPoint ll;
  GPoint ur;
  int nx;
  int ny;
  int nz;
} Probe;


void write_probedata_vtk(double** data, Probe probe, scalar* scalar_fields, vector* vector_fields, char* path){
  int nrows = 0;
  if (scalar_fields) for (scalar field = *scalar_fields, *_i82 = scalar_fields; ((scalar *)&field)->i >= 0; field = *++_i82){
    nrows += 1;
  }

  if (vector_fields) for (vector field = *vector_fields, *_i83 = vector_fields; ((scalar *)&field)->i >= 0; field = *++_i83){
    nrows += 3;
  }

  int ncols = probe.nx*probe.ny*probe.nz;

  const double dx = (probe.ur.x - probe.ll.x)/(probe.nx-1);
  const double dy = (probe.ur.y - probe.ll.y)/(probe.ny-1);
  const double dz = (probe.ur.z - probe.ll.z)/(probe.nz-1);

  double x, y, z;

  FILE* fp;
  fp = fopen(path, "w");

  fputs ("# vtk DataFile Version 2.0\n"
         "Basilisk\n"
         "ASCII\n"
         "DATASET STRUCTURED_GRID\n", fp);
  fprintf(fp, "DIMENSIONS %d %d %d\n", probe.nx, probe.ny, probe.nz);
  fprintf(fp, "POINTS %d double\n", ncols);

  for(int k = 0; k < probe.nz; k++){
    z = probe.ll.z + k*dz;
    for(int j = 0; j < probe.ny; j++){
      y = probe.ll.y + j*dy;
      for(int i = 0; i < probe.nx; i++){
        x = probe.ll.x + i*dx;

        fprintf (fp, "%g %g %g\n", x, y, z);
      }
    }
  }

  fprintf (fp, "POINT_DATA %d\n", ncols);

  int row_index = 0;
  if (scalar_fields) for (scalar field = *scalar_fields, *_i84 = scalar_fields; ((scalar *)&field)->i >= 0; field = *++_i84){
    fprintf (fp, "SCALARS %s double\n", _attribute[field.i].name);
    fputs ("LOOKUP_TABLE default\n", fp);
    for(int col=0; col<ncols; col++){
      fprintf(fp, "%g\n", data[row_index][col]);
    }
    row_index += 1;
  }

  if (vector_fields) for (vector field = *vector_fields, *_i85 = vector_fields; ((scalar *)&field)->i >= 0; field = *++_i85){
    fprintf (fp, "VECTORS %s double\n", _attribute[field.x.i].name);
    for(int col=0; col<ncols; col++){
      fprintf(fp, "%g %g %g\n", data[row_index][col], data[row_index+1][col], data[row_index+2][col]);
    }
    row_index += 3;
  }
  fflush (fp);
  fclose(fp);
}


void probe_fields(Probe probe, scalar* scalar_fields, vector* vector_fields, char* path){







  int nrows = 0;
  if (scalar_fields) for (scalar field = *scalar_fields, *_i86 = scalar_fields; ((scalar *)&field)->i >= 0; field = *++_i86){
    nrows += 1;
  }

  if (vector_fields) for (vector field = *vector_fields, *_i87 = vector_fields; ((scalar *)&field)->i >= 0; field = *++_i87){
    nrows += 3;
  }

  int ncols = probe.nx*probe.ny*probe.nz;

  double **data = pmalloc(nrows*sizeof(double *),__func__,__FILE__,__LINE__);
  for(int i=0; i<nrows; i++){
    data[i] = pmalloc(ncols*sizeof(double),__func__,__FILE__,__LINE__);
  }

  const double dx = (probe.ur.x - probe.ll.x)/(probe.nx-1);
  const double dy = (probe.ur.y - probe.ll.y)/(probe.ny-1);
  const double dz = (probe.ur.z - probe.ll.z)/(probe.nz-1);

  double x, y, z;

  int row_index = 0;
  int col_index = 0;
  for(int k = 0; k < probe.nz; k++){
    z = probe.ll.z + k*dz;
    for(int j = 0; j < probe.ny; j++){
      y = probe.ll.y + j*dy;
      for(int i = 0; i < probe.nx; i++){
        x = probe.ll.x + i*dx;


        Point point = locate ((struct _locate){x, y, z});  int ig = 0; NOT_UNUSED(ig); int jg = 0; NOT_UNUSED(jg); int kg = 0; NOT_UNUSED(kg); POINT_VARIABLES; 
#line 129 "probing.c"


        if (scalar_fields) for (scalar field = *scalar_fields, *_i88 = scalar_fields; ((scalar *)&field)->i >= 0; field = *++_i88){
          data[row_index][col_index] = (point.level >= 0) ? val(field,0,0,0) : nodata;
          row_index += 1;
        }

        if (vector_fields) for (vector field = *vector_fields, *_i89 = vector_fields; ((scalar *)&field)->i >= 0; field = *++_i89){

          data[row_index][col_index] = (point.level >= 0) ? val(field.x,0,0,0) : nodata;
          row_index += 1;

          data[row_index][col_index] = (point.level >= 0) ? val(field.y,0,0,0) : nodata;
          row_index += 1;

          data[row_index][col_index] = (point.level >= 0) ? val(field.z,0,0,0) : nodata;
          row_index += 1;
        }

        row_index = 0;
        col_index += 1;
      }
    }
  }



  if (pid() == 0){
    for(int row=0; row<nrows; row++){
      MPI_Reduce(MPI_IN_PLACE, data[row], ncols, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    }
  }
  else{
    for(int row=0; row<nrows; row++){
      MPI_Reduce(data[row], data[row], ncols, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    }
  }



  if(pid() == 0){
    write_probedata_vtk(data, probe, scalar_fields, vector_fields, path);
  }

  for(int i=0; i<nrows; i++){
    pfree(data[i],__func__,__FILE__,__LINE__);
  }
  pfree(data,__func__,__FILE__,__LINE__);
}


void iprobe_fields(Probe probe, scalar* scalar_fields, vector* vector_fields, char* path){






  int nrows = 0;
  if (scalar_fields) for (scalar field = *scalar_fields, *_i90 = scalar_fields; ((scalar *)&field)->i >= 0; field = *++_i90){
    nrows += 1;
  }

  if (vector_fields) for (vector field = *vector_fields, *_i91 = vector_fields; ((scalar *)&field)->i >= 0; field = *++_i91){
    nrows += 3;
  }

  int ncols = probe.nx*probe.ny*probe.nz;

  double **data = pmalloc(nrows*sizeof(double *),__func__,__FILE__,__LINE__);
  for(int i=0; i<nrows; i++){
    data[i] = pmalloc(ncols*sizeof(double),__func__,__FILE__,__LINE__);
  }

  const double dx = (probe.ur.x - probe.ll.x)/(probe.nx-1);
  const double dy = (probe.ur.y - probe.ll.y)/(probe.ny-1);
  const double dz = (probe.ur.z - probe.ll.z)/(probe.nz-1);

  double x, y, z, value;

  int row_index = 0;
  int col_index = 0;


  if (scalar_fields) for (scalar field = *scalar_fields, *_i92 = scalar_fields; ((scalar *)&field)->i >= 0; field = *++_i92){

    for(int k = 0; k < probe.nz; k++){
      z = probe.ll.z + k*dz;
      for(int j = 0; j < probe.ny; j++){
        y = probe.ll.y + j*dy;
        for(int i = 0; i < probe.nx; i++){
          x = probe.ll.x + i*dx;

          value = interpolate((struct _interpolate){field, x, y, z});
          data[row_index][col_index] = (value < nodata) ? value : nodata;

          col_index += 1;
        }
      }
    }

    col_index = 0;
    row_index += 1;
  }

  if (vector_fields) for (vector field = *vector_fields, *_i93 = vector_fields; ((scalar *)&field)->i >= 0; field = *++_i93){

    for(int k = 0; k < probe.nz; k++){
      z = probe.ll.z + k*dz;
      for(int j = 0; j < probe.ny; j++){
        y = probe.ll.y + j*dy;
        for(int i = 0; i < probe.nx; i++){
          x = probe.ll.x + i*dx;

          value = interpolate((struct _interpolate){field.x, x, y, z});
          data[row_index][col_index] = (value < nodata) ? value : nodata;

          value = interpolate((struct _interpolate){field.y, x, y, z});
          data[row_index+1][col_index] = (value < nodata) ? value : nodata;

          value = interpolate((struct _interpolate){field.z, x, y, z});
          data[row_index+2][col_index] = (value < nodata) ? value : nodata;

          col_index += 1;
        }
      }
    }

    col_index = 0;
    row_index += 3;
  }



  if (pid() == 0){
    for(int row=0; row<nrows; row++){
      MPI_Reduce(MPI_IN_PLACE, data[row], ncols, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    }
  }
  else{
    for(int row=0; row<nrows; row++){
      MPI_Reduce(data[row], data[row], ncols, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    }
  }



  if(pid() == 0){
    write_probedata_vtk(data, probe, scalar_fields, vector_fields, path);
  }

  for(int i=0; i<nrows; i++){
    pfree(data[i],__func__,__FILE__,__LINE__);
  }
  pfree(data,__func__,__FILE__,__LINE__);
}
#line 295 "probing.c"
vector field= {{0},{1},{2}};
scalar foo= {3};
scalar bar= {4};

int main(){ _init_solver();
  char path[256];

  GPoint ll = {0.2, 0.2, 0.2};
  GPoint ur = {0.8, 0.8, 0.8};
  Probe probe = {ll, ur, 4, 3, 5};

  init_grid(1 << 8);

   { foreach(){

#line 308 "probing.c"
{
    val(field.x,0,0,0) = x;
    val(field.y,0,0,0) = -y;
    val(field.z,0,0,0) = 2*z;

    val(foo,0,0,0) = x + y + z;
    val(bar,0,0,0) = x + 2*y + 3*z;
  } } end_foreach(); }

  sprintf(path, "sine_0.vtk");
  probe_fields(probe, ((scalar []){foo,bar,{-1}}), ((vector []){{field.x,field.y,field.z},{{-1},{-1},{-1}}}), path);

  sprintf(path, "isine_0.vtk");
  iprobe_fields(probe, ((scalar []){foo,bar,{-1}}), ((vector []){{field.x,field.y,field.z},{{-1},{-1},{-1}}}), path);
 free_solver(); }
size_t datasize = 5*sizeof (double);
void _init_solver (void) {
  void init_solver();
  init_solver();
  Events = (Event *) pmalloc (sizeof (Event), __func__, __FILE__, __LINE__);
  Events[0].last = 1;
  _attribute = (_Attributes *) pcalloc (datasize/sizeof(double), sizeof (_Attributes), __func__, __FILE__, __LINE__);
  all = (scalar *) pmalloc (sizeof (scalar)*6,__func__, __FILE__, __LINE__);
  for (int i = 0; i < 5; i++)
    all[i].i = i;
  all[5].i = -1;
  set_fpe();
  octree_methods();
  init_scalar ((scalar){4}, "bar");
  init_scalar ((scalar){3}, "foo");
  init_vector ((vector){{0},{1},{2}}, "field");
  init_const_scalar ((scalar){_NVARMAX+7}, "zeroc",  0.);
  init_const_scalar ((scalar){_NVARMAX+6}, "unity",  1.);
  init_const_vector ((vector){{_NVARMAX+3},{_NVARMAX+4},{_NVARMAX+5}}, "unityf", (double []) {1.,1.,1.});
  init_const_vector ((vector){{_NVARMAX+0},{_NVARMAX+1},{_NVARMAX+2}}, "zerof", (double []) {0.,0.,0.});
}
