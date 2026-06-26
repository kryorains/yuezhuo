#include <stdarg.h>
#include <stdio.h>
#include <time.h>

static int pushed_char = EOF;

static int read_char(void) {
  if (pushed_char != EOF) {
    int c = pushed_char;
    pushed_char = EOF;
    return c;
  }
  return getchar_unlocked();
}

static void unread_char(int c) {
  if (c != EOF) {
    pushed_char = c;
  }
}

static void flush_pushed_char(void) {
  if (pushed_char != EOF) {
    ungetc(pushed_char, stdin);
    pushed_char = EOF;
  }
}

int getint(void) {
  int c = read_char();
  while (c != EOF && c <= ' ') {
    c = read_char();
  }

  int neg = 0;
  if (c == '-' || c == '+') {
    neg = c == '-';
    c = read_char();
  }

  unsigned int value = 0;
  while (c >= '0' && c <= '9') {
    value = value * 10u + (unsigned int)(c - '0');
    c = read_char();
  }
  unread_char(c);

  return neg ? (int)(0u - value) : (int)value;
}

int getch(void) {
  return read_char();
}

float getfloat(void) {
  float x;
  flush_pushed_char();
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
  char buf[16];
  int len = 0;
  unsigned int value;
  if (x < 0) {
    putchar_unlocked('-');
    value = 0u - (unsigned int)x;
  } else {
    value = (unsigned int)x;
  }
  do {
    buf[len++] = (char)('0' + value % 10u);
    value /= 10u;
  } while (value != 0);
  while (len != 0) {
    putchar_unlocked(buf[--len]);
  }
}

void putch(int x) {
  putchar_unlocked(x);
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
