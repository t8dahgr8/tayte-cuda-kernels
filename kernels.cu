// Tayte Choudhury — CUDA numerical kernels and reproducible benchmarks.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do {cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}while(0)
#define BL(x) do {if((x)!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS error\n");exit(1);}}while(0)
// Online (max,sum) reduction; two global reads, one write. Not a one-read kernel.
__global__ void softmax(const float* x,float* y,int n){
  __shared__ float maxima[256], sums[256];
  int t=threadIdx.x,base=blockIdx.x*n; float m=-INFINITY,s=0;
  for(int j=t;j<n;j+=256){float v=x[base+j], next=fmaxf(m,v);s=s*expf(m-next)+expf(v-next);m=next;}
  maxima[t]=m;sums[t]=s;__syncthreads();
  for(int d=128;d;d/=2){if(t<d){float a=maxima[t],b=maxima[t+d],z=fmaxf(a,b);sums[t]=sums[t]*expf(a-z)+sums[t+d]*expf(b-z);maxima[t]=z;}__syncthreads();}
  for(int j=t;j<n;j+=256)y[base+j]=expf(x[base+j]-maxima[0])/sums[0];
}
__global__ void naive(const float*a,const float*b,float*c,int n){int r=blockIdx.y*16+threadIdx.y,col=blockIdx.x*16+threadIdx.x;if(r<n&&col<n){float s=0;for(int k=0;k<n;k++)s+=a[r*n+k]*b[k*n+col];c[r*n+col]=s;}}
__global__ void tiled(const float*a,const float*b,float*c,int n){
 __shared__ float A[16][16],B[16][16];int x=threadIdx.x,y=threadIdx.y,r=blockIdx.y*16+y,col=blockIdx.x*16+x;float s=0;
 for(int k=0;k<n;k+=16){A[y][x]=(r<n&&k+x<n)?a[r*n+k+x]:0;B[y][x]=(col<n&&k+y<n)?b[(k+y)*n+col]:0;__syncthreads();for(int j=0;j<16;j++)s+=A[y][j]*B[j][x];__syncthreads();}if(r<n&&col<n)c[r*n+col]=s;
}
__global__ void blocked(const float*a,const float*b,float*c,int n){
 __shared__ float A[64][16],B[16][64];int x=threadIdx.x,y=threadIdx.y,r=blockIdx.y*64+y*4,col=blockIdx.x*64+x*4;float v[4][4]={};int tid=y*16+x;
 for(int k=0;k<n;k+=16){for(int q=tid;q<1024;q+=256){int ar=blockIdx.y*64+q/16,ak=k+q%16,bk=k+q/64,bc=blockIdx.x*64+q%64;A[q/16][q%16]=(ar<n&&ak<n)?a[ar*n+ak]:0;B[q/64][q%64]=(bk<n&&bc<n)?b[bk*n+bc]:0;}__syncthreads();
 for(int z=0;z<16;z++)for(int i=0;i<4;i++)for(int j=0;j<4;j++)v[i][j]+=A[y*4+i][z]*B[z][x*4+j];__syncthreads();}
 for(int i=0;i<4;i++)for(int j=0;j<4;j++)if(r+i<n&&col+j<n)c[(r+i)*n+col+j]=v[i][j];
}
template<class F> float timing(F f,int reps=20){for(int i=0;i<3;i++)f();CK(cudaDeviceSynchronize());cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));CK(cudaEventRecord(a));for(int i=0;i<reps;i++)f();CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));return ms/reps;}
int main(){cudaDeviceProp prop;CK(cudaGetDeviceProperties(&prop,0));fprintf(stderr,"GPU: %s; times exclude transfers, 3 warmups, 20 repetitions\n",prop.name);printf("kernel,rows,columns,ms,GB_s,GFLOP_s,max_abs_error\n");
 for(int n:{128,512,2048,8192,65536}){int rows=std::max(32,1048576/n);size_t count=size_t(rows)*n;std::vector<float>x(count),out(count);for(size_t i=0;i<count;i++)x[i]=sinf(float(i)*.13f)*8;float *a,*b;CK(cudaMalloc(&a,count*4));CK(cudaMalloc(&b,count*4));CK(cudaMemcpy(a,x.data(),count*4,cudaMemcpyHostToDevice));auto f=[&](){softmax<<<rows,256>>>(a,b,n);};float ms=timing(f);CK(cudaMemcpy(out.data(),b,count*4,cudaMemcpyDeviceToHost));double err=0;for(int r=0;r<rows;r++){double m=-INFINITY,s=0;for(int j=0;j<n;j++)m=std::max(m,double(x[r*n+j]));for(int j=0;j<n;j++)s+=exp(x[r*n+j]-m);for(int j=0;j<n;j++)err=std::max(err,fabs(out[r*n+j]-exp(x[r*n+j]-m)/s));}if(err>1e-5)return 2;printf("online_softmax,%d,%d,%.6f,%.3f,0,%.9g\n",rows,n,ms,count*8/ms/1e6,err);CK(cudaFree(a));CK(cudaFree(b));}
 cublasHandle_t h;BL(cublasCreate(&h));BL(cublasSetMathMode(h,CUBLAS_PEDANTIC_MATH));
 for(int n:{65,256,2048}){size_t count=size_t(n)*n;std::vector<float>a(count),b(count),ref(count),out(count);for(size_t i=0;i<count;i++){a[i]=sinf(float(i)*.12f);b[i]=cosf(float(i)*.17f);}float *A,*B,*C,*R;CK(cudaMalloc(&A,count*4));CK(cudaMalloc(&B,count*4));CK(cudaMalloc(&C,count*4));CK(cudaMalloc(&R,count*4));CK(cudaMemcpy(A,a.data(),count*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(B,b.data(),count*4,cudaMemcpyHostToDevice));float one=1,zero=0;auto lib=[&](){BL(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&one,B,n,A,n,&zero,R,n));};float lm=timing(lib);CK(cudaMemcpy(ref.data(),R,count*4,cudaMemcpyDeviceToHost));printf("cublas,%d,%d,%.6f,0,%.3f,0\n",n,n,lm,2.0*n*n*n/lm/1e6);
 for(int kind=0;kind<3;kind++){auto f=[&](){dim3 t(16,16),g((n+15)/16,(n+15)/16),g4((n+63)/64,(n+63)/64);if(kind==0)naive<<<g,t>>>(A,B,C,n);if(kind==1)tiled<<<g,t>>>(A,B,C,n);if(kind==2)blocked<<<g4,t>>>(A,B,C,n);};float ms=timing(f);CK(cudaMemcpy(out.data(),C,count*4,cudaMemcpyDeviceToHost));double err=0;for(size_t i=0;i<count;i++)err=std::max(err,fabs(double(out[i]-ref[i])));if(err>0.01)return 3;printf("%s,%d,%d,%.6f,0,%.3f,%.9g\n",kind==0?"naive":kind==1?"tiled":"register_blocked",n,n,ms,2.0*n*n*n/ms/1e6,err);}CK(cudaFree(A));CK(cudaFree(B));CK(cudaFree(C));CK(cudaFree(R));}BL(cublasDestroy(h));
}
