#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>

#include <pthread.h>
#include <semaphore.h>


#include <syslog.h>         /* for writing to syslog */
#include <time.h>           /* for timespec */
#include <sys/time.h>       /* for timespec */
#include <sched.h>          /* for scheduling */


#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define HRES 640
#define VRES 480
#define MY_CLOCK_TYPE CLOCK_MONOTONIC_RAW
#define NANOSEC_PER_SEC (1000000000)
#define USEC_PER_MSEC (1000)

#define NUM_CPU_CORES 4
#define TRUE 1
#define FALSE 0

#define TOTAL_FRAMES 10
#define ITEM_SIZE (HRES*VRES*3)
#define MAX_BUFFER_SIZE 40

#define SERVICE_1_PERIOD 3
#define SERVICE_2_PERIOD 9

sem_t read_sem, write_sem;
int read_done,write_done=FALSE;

pthread_mutex_t buff_mutex;

double start_time;
unsigned long long seqCnt;
int mat_type;

double getTimeMsec(void);
unsigned int millis ();


/*********************************************************
* Ring Buffer functions
**********************************************************/
//Create Ring Buffer Structure
struct ringBuffer* buf;
struct ringBuffer {
    int status;
    int index_push;
    int index_pull;
    int new_items;
    double read_time [MAX_BUFFER_SIZE];
    unsigned char* buffer;
};

//Initialize the Ring Buffer
void ringBuffer_Init(struct ringBuffer *buf) {
    buf -> status = -1;  //-1 not started yet,0=empty, 1=OK, 2=full
    buf -> index_push = 0;
    buf -> index_pull = 0;
    buf -> new_items = 0;
    for (int i=1;i < MAX_BUFFER_SIZE; i++)
      buf->read_time[i] = (double)(0);
    buf -> buffer = (unsigned char*)calloc(MAX_BUFFER_SIZE, ITEM_SIZE);
    return;
}

void ringBuffer_Push(struct ringBuffer *buf, Mat frame) {
  //cout <<"Pushing Frame: "<<std::to_string(buf -> index_push)<<"\n";
  pthread_mutex_lock(&buff_mutex);
  double push_time = getTimeMsec()-start_time;
  memcpy(buf -> buffer + (buf -> index_push) * (ITEM_SIZE), frame.data, ITEM_SIZE);
  buf->read_time[buf->index_push] = push_time;
  buf -> new_items++;
  buf -> index_push = (buf -> index_push + 1) % (MAX_BUFFER_SIZE);
  if (buf -> new_items < MAX_BUFFER_SIZE)
    buf -> status = 1;
  else
    buf -> status = 2;
  pthread_mutex_unlock(&buff_mutex);
}

int ringBuffer_Pull(struct ringBuffer *buf,int mat_type,Mat frame ) {
  pthread_mutex_lock(&buff_mutex);
  //cout <<"Pulling Frame: "<<std::to_string(buf -> index_pull)<<"\n";
  if (buf -> new_items == 0) {
    buf->status = 0;
    pthread_mutex_unlock(&buff_mutex);
    return -1;
  }

  //cout <<"Frame Written: "<<setprecision(6)<<push_time<<" Frame Read: "<<setprecision(6)<<pull_time<<"\n";
  memcpy(frame.data, buf -> buffer + (buf -> index_pull) * (ITEM_SIZE), ITEM_SIZE*sizeof(unsigned char));
  buf -> index_pull = (buf -> index_pull + 1) % (MAX_BUFFER_SIZE);
  buf -> new_items--;
  if (buf->new_items < MAX_BUFFER_SIZE)
    buf->status = 1;
  pthread_mutex_unlock(&buff_mutex);
  return 0;
}

string ringBuffer_Status(struct ringBuffer *buf) {
  string out_status = "None";
  switch(buf->status) {
    case -1 :
      out_status = "No Data";
      break;
    case 0:
      out_status = "Empty";
      break;
    case 1:
      out_status = "OK";
      break;
    case 2:
      out_status = "Full";
      break;
    default:
      out_status = "Unkown";
      break;
  }
  return out_status;
}

/*********************************************************
* Read Frame Thread
**********************************************************/
void *readFrames(void* arg)
{
    int cnt=0;
    double thread_start = -1;
    double thread_time = 0;
    VideoCapture cap;
    Mat frame;

    start_time = getTimeMsec();
    syslog (LOG_DEBUG,"--------------------------------------------");
    syslog (LOG_DEBUG,"                       Freq (Hz)");
    syslog (LOG_DEBUG,"Action      Cnt      Read    Write    Buffer");
    syslog (LOG_DEBUG,"--------------------------------------------");
    //syslog (LOG_DEBUG,"Capt Start  %03d     %03d",cnt,(int)((getTimeMsec()-start_time)/USEC_PER_MSEC*1000));
    //cout << "Capturer thread's running on CPU=" << sched_getcpu() << ". \n";

    cap.open(5);
    if (!cap.isOpened())
    {
        cerr << "ERROR! Unable to open camera\n";
        return 0;
    }

    // Set resolution
    cap.set(CAP_PROP_FRAME_WIDTH, HRES);
    cap.set(CAP_PROP_FRAME_HEIGHT, VRES);



    cap.read(frame);
    mat_type = frame.type();

    cout <<"Started capturing frames\n";

    do
    {
        sem_wait(&read_sem);
        cap.read(frame);

        if (thread_start == -1 ) {
          thread_start = getTimeMsec();
        }



        if(frame.empty())
        {
            cerr << "ERROR: No image grabbed\n";
            return 0;
        }


        ringBuffer_Push(buf,frame);


        thread_time = getTimeMsec()-thread_start;
        thread_start = getTimeMsec();
        syslog (LOG_DEBUG,"Read        %03d     %03d             %s",cnt,(int)(1000/thread_time),ringBuffer_Status(buf).c_str());

        cnt++;


    } while (cnt < TOTAL_FRAMES);

    read_done = TRUE;
    pthread_exit((void *)0);
}

/*********************************************************
* Write Frame Thread
**********************************************************/
void *writeFrames(void* arg)
{
    int cnt=0;
    double thread_start = -1;
    double thread_time = 0;

    do {

        sem_wait(&write_sem);

        if (thread_start == -1 ) {
          thread_time = 1000000;
          thread_start = getTimeMsec();
        }
        else {
          thread_time = getTimeMsec()-thread_start;
          thread_start = getTimeMsec();
        }


        if (buf->status != -1) {
            Mat r_frame(VRES, HRES, mat_type);
            int status  = ringBuffer_Pull(buf,mat_type,r_frame);
            if (status != -1)
            {
              stringstream file_name;
              file_name << "./img/Image" << std::to_string(cnt) << ".jpg";
              //cout <<"Outputting File"<< file_name.str() << "\n";

              imwrite(file_name.str(), r_frame);
              cnt++;
            }
            syslog (LOG_DEBUG,"Write       %03d             %03d     %s",cnt,(int)(1000/thread_time),ringBuffer_Status(buf).c_str());
        }

   } while (cnt < TOTAL_FRAMES);


    write_done = TRUE;
    pthread_exit((void *)0);
}

/*********************************************************
* Sequencer Thread
**********************************************************/

void delayCount(void) {
  struct timespec delay_time = {0,100000000}; // delay for 100 msec, 10 Hz
  struct timespec remaining_time;
  double residual;
  int rc,delay_cnt=0;
  delay_cnt=0; residual=0.0;
  do
  {
      rc=nanosleep(&delay_time, &remaining_time);
      if(rc == EINTR)
      {
          residual = remaining_time.tv_sec + ((double)remaining_time.tv_nsec / (double)NANOSEC_PER_SEC);

          if(residual > 0.0) printf("residual=%lf, sec=%d, nsec=%d\n", residual, (int)remaining_time.tv_sec, (int)remaining_time.tv_nsec);

          delay_cnt++;
      }
      else if(rc < 0)
      {
          perror("Sequencer nanosleep");
          exit(-1);
      }



  } while((residual > 0.0) && (delay_cnt < 100));
}


void *sequencer(void *arg)
{
  seqCnt=0;
  cout << "Sequencer running on CPU=" << sched_getcpu() << ". \n";
  do {
      //cout << "Sequencer running frame "<<seqCnt<<"\n";

      if((seqCnt % SERVICE_1_PERIOD) == 0) sem_post(&read_sem);
      if((seqCnt % SERVICE_2_PERIOD) == 0) sem_post(&write_sem);

      delayCount();
      seqCnt ++;

  } while (write_done != TRUE);

  cout << "Sequencer Ending\n ";
  pthread_exit((void *)0);
}

/*********************************************************
* Main Function
**********************************************************/
int main(int argc, char** argv )
{

    int scope,rc;
    unsigned long int  start_millis,stop_millis;

    start_millis=millis();


    cpu_set_t threadcpu,allcpuset;
    pthread_t read_thread, write_thread, seq_thread;
    struct sched_param main_param,seq_param,read_param, write_param;
    pthread_attr_t main_attr, seq_attr,read_attr, write_attr;
    pid_t mainpid;
    int rt_max_prio;


    //Directory for images
    system("rm img -r");
    system("mkdir img");

    // Initialize buffer
    struct ringBuffer buf_obj;
    buf = &buf_obj;
    ringBuffer_Init(buf);

    //Init pthreads
    pthread_mutex_init(&buff_mutex, NULL);
    if (sem_init (&read_sem, 0, 0)) { printf ("Failed to initialize S1 semaphore\n"); exit (-1); }
    if (sem_init (&write_sem, 0, 0)) { printf ("Failed to initialize S2 semaphore\n"); exit (-1); }

    CPU_ZERO(&allcpuset);
    for(int i=0; i < NUM_CPU_CORES; i++)
        CPU_SET(i, &allcpuset);
    cout << "Using CPUS="<<CPU_COUNT(&allcpuset)<<" from total available.\n";

    mainpid=getpid();

    rt_max_prio = sched_get_priority_max(SCHED_FIFO);

    rc=sched_getparam(mainpid, &main_param);
    main_param.sched_priority=rt_max_prio;
    rc=sched_setscheduler(getpid(), SCHED_FIFO, &main_param);
    if(rc < 0)
      cerr << "ERROR! main_param\n";
    pthread_attr_getscope(&main_attr, &scope);

    //Init Sequencer Thread
    CPU_ZERO(&threadcpu);
    CPU_SET(0, &threadcpu);

    //Init Sequencer Thread
    rc=pthread_attr_init(&seq_attr);
    rc=pthread_attr_setinheritsched(&seq_attr, PTHREAD_EXPLICIT_SCHED);
    rc=pthread_attr_setschedpolicy(&seq_attr, SCHED_FIFO);
    rc=pthread_attr_setaffinity_np(&seq_attr, sizeof(cpu_set_t), &threadcpu);
    seq_param.sched_priority=rt_max_prio;
    pthread_attr_setschedparam(&seq_attr, &seq_param);
    rc=pthread_create(&seq_thread, &seq_attr, sequencer, NULL);
    if(rc < 0)
        cerr << "ERROR! pthread_create for service 0\n";



    //Init Read Thread
    CPU_ZERO(&threadcpu);
    CPU_SET(1, &threadcpu);
    rc=pthread_attr_init(&read_attr);
    rc=pthread_attr_setinheritsched(&read_attr, PTHREAD_EXPLICIT_SCHED);
    rc=pthread_attr_setschedpolicy(&read_attr, SCHED_FIFO);
    rc=pthread_attr_setaffinity_np(&read_attr, sizeof(cpu_set_t), &threadcpu);
    read_param.sched_priority=rt_max_prio-1;
    pthread_attr_setschedparam(&read_attr, &read_param);
    rc=pthread_create(&read_thread, &read_attr, readFrames, NULL);
    if(rc < 0)
        cerr << "ERROR! pthread_create for service 1\n";


    //Init Write Thread
    CPU_ZERO(&threadcpu);
    CPU_SET(2, &threadcpu);
    rc=pthread_attr_init(&write_attr);
    rc=pthread_attr_setinheritsched(&write_attr, PTHREAD_EXPLICIT_SCHED);
    rc=pthread_attr_setschedpolicy(&write_attr, SCHED_FIFO);
    rc=pthread_attr_setaffinity_np(&write_attr, sizeof(cpu_set_t), &threadcpu);
    write_param.sched_priority=rt_max_prio-2;
    pthread_attr_setschedparam(&write_attr, &write_param);
    rc=pthread_create(&write_thread, &write_attr, writeFrames, NULL);
    if(rc < 0)
        cerr << "ERROR! pthread_create for service 2\n";



    pthread_join(read_thread, NULL);
    pthread_join(write_thread, NULL);
    pthread_join(seq_thread, NULL);

    if(pthread_mutex_destroy(&buff_mutex) != 0)
      perror("mutex destroy");

    stop_millis=millis();
    printf("Total execution time: %f s\n",(double)(start_millis-stop_millis)/1000);
    return true;
}

/*********************************************************
* Helper Function
**********************************************************/

unsigned int millis () {
  struct timespec t ;
  clock_gettime ( CLOCK_MONOTONIC_RAW , & t ) ; // change CLOCK_MONOTONIC_RAW to CLOCK_MONOTONIC on non linux computers
  return t.tv_sec * 1000 + ( t.tv_nsec + 500000 ) / 1000000 ;
}


double getTimeMsec(void)
{
  struct timespec event_ts = {0, 0};

  clock_gettime(CLOCK_MONOTONIC, &event_ts);
  return ((event_ts.tv_sec)*1000.0) + ((event_ts.tv_nsec)/1000000.0);
}
