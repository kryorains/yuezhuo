#include <stdarg.h>
#include <stdio.h>
#include <time.h>

int getint(void) {
  int x;
  return scanf("%d", &x) == 1 ? x : 0;
}

int getch(void) {
  return getchar();
}

float getfloat(void) {
  float x;
  return scanf("%f", &x) == 1 ? x : 0.0f;
}

int getarray(int a[]) {
  int n = getint();
  for (int i = 0; i < n; ++i) {
    a[i] = getint();
  }
  return n;
}

int getfarray(float a[]) {
  int n = getint();
  for (int i = 0; i < n; ++i) {
    a[i] = getfloat();
  }
  return n;
}

void putint(int x) {
  printf("%d", x);
}

void putch(int x) {
  putchar(x);
}

void putfloat(float x) {
  printf("%a", x);
}

void putarray(int n, int a[]) {
  printf("%d:", n);
  for (int i = 0; i < n; ++i) {
    printf(" %d", a[i]);
  }
  putchar('\n');
}

void putfarray(int n, float a[]) {
  printf("%d:", n);
  for (int i = 0; i < n; ++i) {
    printf(" %a", a[i]);
  }
  putchar('\n');
}

void putf(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vprintf(fmt, ap);
  va_end(ap);
}

void starttime(void) {}
void stoptime(void) {}
