#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <cmath>
#include <mpi/mpi.h>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include "consts.h"

using namespace std;

pthread_t threads[2];
pthread_mutex_t mutex;
int* tasks;
//ofstream* LogFiles;

double SummaryDisbalance = 0;
bool isFinishedExecution = false;

int ProcessCount;
int ProcessRank;
int RemainingTasks;
int ExecutedTasks;
int AdditionalTasks;
double globalRes = 0;

void SetColor(const string& color) {
    cout << color;
}

void printTasks(int *taskSet) {
    cout << ANSI_RED << "Process :" << ProcessRank;
    for (int i = 0; i < TASK_COUNT; i++) {
        cout << taskSet[i] << " ";
    }
    cout << ANSI_RESET << endl;
}

void initializeTaskSet(int *taskSet, int taskCount) {
    for (int i = 0; i < taskCount; i++) {
        taskSet[i] = 10000 + rand() % (50000 - 10000 + 1);          //задаем "вес"
    }
}

void executeTaskSet(const int* taskSet) {
    for (int i = 0; i < RemainingTasks; i++) {
        pthread_mutex_lock(&mutex);
        int weight = taskSet[i];
        pthread_mutex_unlock(&mutex);

        for (int j = 0; j < weight; j++) {
            globalRes += sqrt(j);
        }

        ExecutedTasks++;
    }
    RemainingTasks = 0;
}

void* ExecutorStartRoutine(void* args) {
    tasks = new int[TASK_COUNT];
    double StartTime, FinishTime, IterationDuration, ShortestIteration, LongestIteration;

    for (int i = 0; i < ITERATION_COUNT; i++) {

        MPI_Barrier(MPI_COMM_WORLD);

        StartTime = MPI_Wtime();

        cout << ANSI_RED << "Iteration " << i << ". Initializing tasks. " << ANSI_RESET << endl;
        initializeTaskSet(tasks, TASK_COUNT);
        ExecutedTasks = 0;
        RemainingTasks = TASK_COUNT;
        AdditionalTasks = 0;

        executeTaskSet(tasks);
        cout << ANSI_BLUE << "Process " << ProcessRank << " executed tasks in " <<
                  MPI_Wtime() - StartTime << " Now requesting for some additional. " << ANSI_RESET << endl;
        int ThreadResponse;

        for (int procIdx = 0; procIdx < ProcessCount; procIdx++) {

            if (procIdx != ProcessRank) {
                cout << ANSI_WHITE << "Process " << ProcessRank << " is asking " << procIdx <<
                          " for some tasks."<< ANSI_RESET << endl;

                MPI_Send(&ProcessRank, 1, MPI_INT, procIdx, 888, MPI_COMM_WORLD);

                cout << ANSI_PURPLE << "waiting for task count" << ANSI_RESET << endl;

                MPI_Recv(&ThreadResponse, 1, MPI_INT, procIdx, SENDING_TASK_COUNT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                cout << ANSI_WHITE << "Process " << procIdx << " answered " << ThreadResponse << ANSI_RESET << endl;

                if (ThreadResponse != NO_TASKS_TO_SHARE) {
                    AdditionalTasks = ThreadResponse;
                    memset(tasks, 0, TASK_COUNT);

                    cout << ANSI_PURPLE << "waiting for tasks" << ANSI_RESET << endl;

                    MPI_Recv(tasks, AdditionalTasks, MPI_INT, procIdx, SENDING_TASKS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    pthread_mutex_lock(&mutex);
                    RemainingTasks = AdditionalTasks;
                    pthread_mutex_unlock(&mutex);

                    executeTaskSet(tasks);
                }
            }

        }
        FinishTime = MPI_Wtime();
        IterationDuration = FinishTime - StartTime;

        MPI_Allreduce(&IterationDuration, &LongestIteration, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&IterationDuration, &ShortestIteration, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        cout << ANSI_GREEN << "Process " << ProcessRank << " executed " << ExecutedTasks <<
                  " tasks. " << AdditionalTasks << " were additional." << ANSI_RESET << endl;
        cout << ANSI_CYAN << "GlobalRes is " << globalRes << ". Time taken: " << IterationDuration << endl;
        SummaryDisbalance += (LongestIteration - ShortestIteration)/LongestIteration;
        cout << "Max time difference: " << LongestIteration - ShortestIteration  << ANSI_RESET << endl;
        cout << ANSI_PURPLE_BG << "Disbalance rate is " <<
                  ((LongestIteration - ShortestIteration)/ LongestIteration) * 100 << "%" << ANSI_RESET << endl;
//        LogFiles[ProcessRank] << IterationDuration << endl;
    }

    cout << ANSI_RED << "Proc " << ProcessRank << " finished iterations sending signal" << ANSI_RESET << endl;
    pthread_mutex_lock(&mutex);
    isFinishedExecution = true;
    pthread_mutex_unlock(&mutex);
    int Signal = EXECUTOR_FINISHED_WORK;
    MPI_Send(&Signal, 1, MPI_INT, ProcessRank, 888, MPI_COMM_WORLD);
    delete [] tasks;
    pthread_exit(nullptr);
}

void* ReceiverStartRoutine(void* args) {
    int AskingProcRank, Answer, PendingMessage;
    MPI_Status status;

    MPI_Barrier(MPI_COMM_WORLD);      // синхронизирует все процессы, участвующие в MPI_COMM_WORLD.
                                            // Все процессы должны достичь этой точки перед продолжением выполнения.
    while (!isFinishedExecution) {

        MPI_Recv(&PendingMessage, 1, MPI_INT, MPI_ANY_SOURCE, 888, MPI_COMM_WORLD, &status);

        if (PendingMessage == EXECUTOR_FINISHED_WORK) {
            cout << ANSI_RED << "Executor finished work on proc " << ProcessRank << ANSI_RESET << endl;
        }

        AskingProcRank = PendingMessage;

        pthread_mutex_lock(&mutex);
        cout << ANSI_YELLOW << "Process " << AskingProcRank << " requested tasks. I have " <<
                  RemainingTasks << " tasks now. " << ANSI_RESET << endl;

        if (RemainingTasks >= MIN_TASKS_TO_SHARE) {
            Answer = RemainingTasks / (ProcessCount * 2);               //отправляем половину от имеющихся задач
            RemainingTasks = RemainingTasks / (ProcessCount * 2);

            cout << ANSI_PURPLE << "Sharing " << Answer << " tasks. " << ANSI_RESET << endl;

            MPI_Send(&Answer, 1, MPI_INT, AskingProcRank, SENDING_TASK_COUNT, MPI_COMM_WORLD);
            MPI_Send(&tasks[TASK_COUNT - Answer], Answer, MPI_INT, AskingProcRank, SENDING_TASKS, MPI_COMM_WORLD);

        } else {
            Answer = NO_TASKS_TO_SHARE;
            MPI_Send(&Answer, 1, MPI_INT, AskingProcRank, SENDING_TASK_COUNT, MPI_COMM_WORLD);
        }
        pthread_mutex_unlock(&mutex);

    }

    pthread_exit(nullptr);
}


int main(int argc, char* argv[]) {
    int ThreadSupport;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &ThreadSupport);     //указываем что все потоки могут обращаться к функциям MPI
    if (ThreadSupport != MPI_THREAD_MULTIPLE) {     //если ошибка
        MPI_Finalize();
        return -1;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &ProcessRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcessCount);

    pthread_mutex_init(&mutex, nullptr);    //используем дефолтные аттрибуты
    pthread_attr_t ThreadAttributes;
    pthread_attr_init(&ThreadAttributes);

//    LogFiles = new ofstream [ProcessCount];
//    char* name = new char[12];
//    for (int i = 0; i < ProcessCount; i++) {
//        sprintf(name, "Log_%d.txt", i);
//        LogFiles[i].open(name);
//    }

    pthread_attr_setdetachstate(&ThreadAttributes, PTHREAD_CREATE_JOINABLE);    //Потоки, созданные с помощью attr, будут созданы в состоянии объединения.


    double start = MPI_Wtime();

    pthread_create(&threads[0], &ThreadAttributes, ReceiverStartRoutine, NULL);
    pthread_create(&threads[1], &ThreadAttributes, ExecutorStartRoutine, NULL);


    pthread_join(threads[0], nullptr);       //  Функция pthread_join() блокирует вызывающий поток, пока указанный поток не завершится.
                                                            //  Указанный поток должен принадлежать текущему процессу и не должен быть отделен.
    pthread_join(threads[1], nullptr);

    pthread_attr_destroy(&ThreadAttributes);
    pthread_mutex_destroy(&mutex);

    if (ProcessRank == 0) {
        cout << ANSI_GREEN << "Summary disbalance:" << SummaryDisbalance / (ITERATION_COUNT) * 100 << "%" << ANSI_GREEN << endl;
        cout << ANSI_GREEN << "time taken: " << MPI_Wtime() - start << ANSI_GREEN << endl;
    }

    MPI_Finalize();
    return 0;
}
