// Distributed two-dimensional Discrete FFT transform
// YOUR NAME HERE
// ECE8893 Project 1


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include <math.h>
#include <mpi.h>

#include "Complex.h"
#include "InputImage.h"

using namespace std;

void Transform1D(Complex* h, int w, Complex* H);

int _2d21d(int i, int j, int w) {
	return i * w + j;
}

void store_transpose(Complex *tempImg, Complex *buf, int N, int w, int rank) {
	for (int i = 0; i < N; i++) {
		int ind = _2d21d(i % w, i / w + rank * N / w, w);
		tempImg[ind] = tempImg[ind] + buf[i];
	}
}

void Transform2D(const char* inputFN) 
{ // Do the 2D transform here.
  // 1) Use the InputImage object to read in the Tower.txt file and
  //    find the width/height of the input image.
  // 2) Use MPI to find how many CPUs in total, and which one
  //    this process is
  // 3) Allocate an array of Complex object of sufficient size to
  //    hold the 2d DFT results (size is width * height)
  // 4) Obtain a pointer to the Complex 1d array of input data
  // 5) Do the individual 1D transforms on the rows assigned to your CPU
  // 6) Send the resultant transformed values to the appropriate
  //    other processors for the next phase.
  // 6a) To send and receive columns, you might need a separate
  //     Complex array of the correct size.
  // 7) Receive messages from other processes to collect your columns
  // 8) When all columns received, do the 1D transforms on the columns
  // 9) Send final answers to CPU 0 (unless you are CPU 0)
  //   9a) If you are CPU 0, collect all values from other processors
  //       and print out with SaveImageData().
  InputImage image(inputFN);  // Create the helper object for reading the image
  int w = image.GetWidth(), h = image.GetHeight();
  // Step (1) in the comments is the line above.
  // Your code here, steps 2-9
  // Step (2)
  int numtasks, rank, rc;
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf ("numtasks=%d##rank=%d\n", numtasks, rank);
  int N = h / numtasks * w;  // the num of elements for each process
  // Step (3)
  Complex outImg[N];
  // Step (4)
  Complex* inImg = image.GetImageData();
  // Step (5)
  for (int i = rank * N, j = 0; i < (rank + 1) * N; i += w, j += w) {
  	Transform1D(inImg + i, w, outImg + j);
  }
  // Step (6)
  if (rank != 0) {
    rc = MPI_Send(outImg, N * sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD);  // all block-send to 0
    if (rc != MPI_SUCCESS) {
    	cout << "Send row 1d fft result failed: rank=" << rank << endl;
    	MPI_Finalize();
        return;
    }
    // column-wise fft
    Complex buf[N];
    MPI_Status status;
 	rc = MPI_Recv(buf, N * sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
	if (rc != MPI_SUCCESS) {
	    cout << "Receive row 1d fft result failed: src=" << 0 << endl;
	    MPI_Finalize();
	    return;
	}
	cout << "column-wise array received from 0: rank=" << rank << endl;
	
	// fft
	Complex outImg2[N];
	for (int i = 0; i < N; i += w) {
		Transform1D(buf + i, w, outImg2 + i);
	}	
	cout << "column-wise fft done: rank=" << rank << endl;

	// send the result
    rc = MPI_Send(outImg2, N * sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD);  // all block-send to 0
    if (rc != MPI_SUCCESS) {
    	cout << "Send col 1d fft result failed: rank=" << rank << endl;
    	MPI_Finalize();
        return;
    }	
  }
  else {
  	Complex tempImg[w * h];
  	// transpose 0's 1d row result
	store_transpose(tempImg, outImg, N, w, 0);
	// step (7)
  	int count = 0;
  	Complex buf[N];
  	while (count < numtasks - 1) {
	  MPI_Status status;
	  rc = MPI_Recv(buf, N * sizeof(Complex), MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
	  if (rc != MPI_SUCCESS) {
	      cout << "Receive row 1d fft result failed: src=" << status.MPI_SOURCE << endl;
	      MPI_Finalize();
	      return;
	  }
	  cout << "received from " << status.MPI_SOURCE << endl;
	  count++;
	  store_transpose(tempImg, buf, N, w, status.MPI_SOURCE);
  	}
  	// send to others (do not need blocking)
  	for (int r = 1; r < numtasks; r++) {
	    rc = MPI_Send(tempImg + r * N, N * sizeof(Complex), MPI_CHAR, r, 0, MPI_COMM_WORLD);  // all block-send to 0
	    if (rc != MPI_SUCCESS) {
	    	cout << "Send row 1d fft result failed: rank=" << r << endl;
	    	MPI_Finalize();
	        return;
	    }
	    cout << "sent to " << r << endl;
    }
    // column-wise fft
    Complex outImg2[N];
    for (int i = 0; i < N; i += w) {
  		Transform1D(tempImg + i, w, outImg2 + i);
  	}
  	// store self result
  	Complex finalImg[h * w];
  	store_transpose(finalImg, outImg2, N, w, 0);
  	// collect others' result
  	count = 0;
  	while (count < numtasks - 1) {
	  MPI_Status status;
	  rc = MPI_Recv(buf, N * sizeof(Complex), MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
	  if (rc != MPI_SUCCESS) {
	      cout << "Receive row 1d fft result failed: src=" << status.MPI_SOURCE << endl;
	      MPI_Finalize();
	      return;
	  }
	  count++;
	  store_transpose(finalImg, buf, N, w, status.MPI_SOURCE);
  	}
  	// save image
  	image.SaveImageData("2d.out.text", finalImg, w, h);
  }
}

void Transform1D(Complex* h, int w, Complex* H)
{
  // Implement a simple 1-d DFT using the double summation equation
  // given in the assignment handout.  h is the time-domain input
  // data, w is the width (N), and H is the output array.
  double coef = 2.0 * M_PI / w;
  for (int n = 0; n < w; n++) {
  	for (int k = 0; k < w; k++) {
  	  H[n] = H[n] + h[k] * Complex(cos(coef * n * k), -sin(coef * n * k));
  	}
  }
}

int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line

  // MPI initialization here 
  int rc = MPI_Init(&argc,&argv);
  if (rc != MPI_SUCCESS) {
    printf ("Error starting MPI program. Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }

  Transform2D(fn.c_str()); // Perform the transform.

  // Finalize MPI here
  MPI_Finalize();
}  
  

  
