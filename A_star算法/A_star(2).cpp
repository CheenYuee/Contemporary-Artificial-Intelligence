#include <iostream>
#include <queue>
#include <vector>
#include <cmath>
using namespace std;

const int unlinked = 1000;

// 状态类
class State_Info
{
public:
    int state;
    int cost_g;
    int cost_h;
    int cost_f;

    State_Info(int input_state, int g)
    {
        this->state = input_state;
        this->cost_g = g;
        this->cost_h = 0;
        this->cost_f = cost_g + cost_h;
    }

    void compute_f(int M, int route[][3])
    {
        if(state == 1)
        {
            return;
        }
        int Min = unlinked;
        for(int i = 0; i < M; i++)
        {
            if(route[i][0] == state && route[i][2] < Min)
            {
                Min = route[i][2];
            }
        }
        this->cost_h = Min;
        this->cost_f = cost_g + cost_h;
    }

    vector<int> generate_next(int M, int route[][3])
    {
        vector<int> next;
        for(int i = 0; i < M; i++)
        {
            if(route[i][0] == state)
            {
                next.push_back(route[i][1]);
            }
        }
        return next;
    }
};

void A_star_algorithm(int N, int M, int K, int route[][3])
{

    // 邻接矩阵
    int matrix[N+1][N+1];
    for(int i = 0; i <= N; i++)
    {
        for(int j = 0; j <= N; j++)
        {
            matrix[i][j] = unlinked;
        }
    }
    for(int i = 0; i < M; i++)
    {
        matrix[ route[i][0] ][ route[i][1] ] = route[i][2];
    }

    vector<int> shortest_cost;
    priority_queue <pair<int, State_Info*>, vector<pair<int, State_Info*> >, greater<pair<int, State_Info*> > > q;
    State_Info* tmp = new State_Info(N, 0);
    q.push(make_pair(tmp->cost_f, tmp));

    int n = 0;
    while(!q.empty() && shortest_cost.size() < K)
    {
        q.pop();
        //cout << "state:" << tmp->state << endl;
        //cout << "cost_g:" << tmp->cost_g << endl;
        //cout << "cost_f:" << tmp->cost_f << endl;
        if(tmp->state == 1)
        {
            shortest_cost.push_back(tmp->cost_g);
        }
        vector<int> next = tmp->generate_next(M, route);
        for(unsigned i = 0; i < next.size(); i++)
        {
            int cost_g_next = tmp->cost_g + matrix[tmp->state][next[i]];
            State_Info* state_next = new State_Info(next[i], cost_g_next);
            state_next->compute_f(M, route);
            q.push(make_pair(state_next->cost_f, state_next));
        }
        tmp = q.top().second;
    }

    if(q.size() == 0)
    {
        for(int i = K - shortest_cost.size(); i > 0; i--)
        {
            shortest_cost.push_back(-1);
        }
    }
    cout << "Find the state." << endl;
    cout << "cost:" << endl;
    for(unsigned i = 0; i < shortest_cost.size(); i++)
    {
        cout << shortest_cost[i] << endl;
    }

}

int main()
{

    int N0 = 8, M0 = 16, K0 = 16;
    int route0[M0][3] = {{8, 7, 2},
                        {8, 5, 2},
                        {8, 4, 3},
                        {8, 2, 1},
                        {7, 6, 2},
                        {7, 4, 3},
                        {6, 3, 2},
                        {6, 5, 1},
                        {5, 4, 1},
                        {5, 3, 1},
                        {5, 2, 1},
                        {5, 1, 1},
                        {4, 3, 4},
                        {3, 1, 1},
                        {3, 2, 1},
                        {2, 1, 1}};
    cout << "test input:  " << endl;
    int N = 5, M = 8, K = 7;
    int route[M][3];
    cin >> N;
    cin >> M;
    cin >> K;
    for(int i = 0; i < M; i++)
    {
        cin >> route[i][0];
        cin >> route[i][1];
        cin >> route[i][2];
    }
    //cout << "test input:  ";
    A_star_algorithm(N, M, K, route);
    return 0;
}
