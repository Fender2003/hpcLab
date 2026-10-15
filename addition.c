#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main(){
    struct timespec ts;
    int a[50000], b[50000];

    for(int i=0; i<50000; i++){
        a[i] = rand();
        b[i] = rand();
    }
    int c[50000] = {0};

    for(int i=0; i<50000; i++){
        c[i] = a[i] + b[i];
    }

    printf('%d\n', c[0]);
    return 0;
}