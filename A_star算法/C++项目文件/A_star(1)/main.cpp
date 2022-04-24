#include <iostream>
#include <queue>
#include <vector>
#include <map>
#include <cmath>
using namespace std;

string state_goal_str = "123804765";
char state_goal_matrix[3][3] = {{'1', '2', '3'},
                        {'8', '0', '4'},
                        {'7', '6', '5'}};

string state_input_str0 = "283104765";

// 这里定义了5个测试用例
string state_input_str1 = "024657318";
string state_input_str2 = "587346120";
string state_input_str3 = "375148206";
string state_input_str4 = "512768340";
string state_input_str5 = "123804765";

// 状态类
class State_Info
{
public:
    string state_str;
    int cost_g;
    int cost_h;
    int cost_f;
    int zero_i;
    int zero_j;
    char state_matrix[3][3];

    State_Info(string input_str, int g)
    {

        int index = 0;
        for(int i = 0; i < 3; i++)
        {
            for(int j = 0; j < 3; j++)
            {
                state_matrix[i][j] = input_str[index];
                index++;
                if(state_matrix[i][j] == '0')
                {
                    this->zero_i = i;
                    this->zero_j = j;
                }
            }
        }
        this->state_str = input_str;
        this->cost_g = g;
        this->cost_h = compute_h();
        this->cost_f = cost_g + cost_h;

    }
/*
    // 另一种没有采用的启发式函数
    int compute_h2(void)
    {

        int h = 0;

        for(unsigned i = 0; i < state_goal_str.size(); i++)
        {
            if(state_str[i] == '0')
            {
                continue;
            }
            if(state_str[i] != state_goal_str[i])
            {
                h++;
            }
        }
        return h;
    }
*/
    // 启发式函数h
    int compute_h(void)
    {
        int h = 0;
        for(int number = 1; number <= 8; number++)
        {
            int this_i, this_j;
            int goal_i, goal_j;
            for(int i = 0; i < 3; i++)
            {
                for(int j = 0; j < 3; j++)
                {
                    if(state_goal_matrix[i][j] == '0' + number)
                    {
                        goal_i = i;
                        goal_j = j;
                    }
                    if(this->state_matrix[i][j] == '0' + number)
                    {
                        this_i = i;
                        this_j = j;
                    }
                }
            }
            h += abs(this_i - goal_i) + abs(this_j - goal_j);
        }
        return h;
    }
    // 将状态矩阵转换成字符串
    string generate_string(char matrix[3][3])
    {
        string s;
        for(int i = 0; i < 3; i++)
        {
            for(int j = 0; j < 3; j++)
            {
                s.push_back(matrix[i][j]);
            }
        }
        return s;
    }
    // 生成下一步可到达的状态
    vector<string> generate_next(void)
    {
        vector<string> next;

        if(zero_j > 0)
        {
            // 左移
            state_matrix[zero_i][zero_j] = state_matrix[zero_i][zero_j - 1];
            state_matrix[zero_i][zero_j - 1] = '0';
            string s = generate_string(state_matrix);
            next.push_back(s);
            state_matrix[zero_i][zero_j - 1] = state_matrix[zero_i][zero_j];
            state_matrix[zero_i][zero_j] = '0';
        }
        if(zero_j < 2)
        {
            // 右移
            state_matrix[zero_i][zero_j] = state_matrix[zero_i][zero_j + 1];
            state_matrix[zero_i][zero_j + 1] = '0';
            string s = generate_string(state_matrix);
            next.push_back(s);
            state_matrix[zero_i][zero_j + 1] = state_matrix[zero_i][zero_j];
            state_matrix[zero_i][zero_j] = '0';
        }
        if(zero_i > 0)
        {
            // 上移
            state_matrix[zero_i][zero_j] = state_matrix[zero_i - 1][zero_j];
            state_matrix[zero_i - 1][zero_j] = '0';
            string s = generate_string(state_matrix);
            next.push_back(s);
            state_matrix[zero_i - 1][zero_j] = state_matrix[zero_i][zero_j];
            state_matrix[zero_i][zero_j] = '0';
        }
        if(zero_i < 2)
        {
            // 下移
            state_matrix[zero_i][zero_j] = state_matrix[zero_i + 1][zero_j];
            state_matrix[zero_i + 1][zero_j] = '0';
            string s = generate_string(state_matrix);
            next.push_back(s);
            state_matrix[zero_i + 1][zero_j] = state_matrix[zero_i][zero_j];
            state_matrix[zero_i][zero_j] = '0';
        }
        return  next;
    }

};

void A_star_algorithm(string state_input_str)
{
    priority_queue <pair<int, State_Info*>, vector<pair<int, State_Info*> >, greater<pair<int, State_Info*> > > q;
    State_Info* state = new State_Info(state_input_str, 0);
    q.push(make_pair(state->cost_f, state));

    State_Info* tmp = q.top().second;
    map<string, int> visited;
    int n = 0;
    while(q.size() && tmp->state_str != state_goal_str)
    {
        n++;
        q.pop();
        visited[tmp->state_str] = 1;
        vector<string> next = tmp->generate_next();
        for(unsigned i = 0; i < next.size(); i++)
        {
            if(visited.count(next[i]) == 1)  continue;
            State_Info* state_next = new State_Info(next[i], tmp->cost_g + 1);
            q.push(make_pair(state_next->cost_f, state_next));
        }
        tmp = q.top().second;
    }
    if(q.size() == 0)
    {
        cout << "Don't find the state." << endl;
    }else{
        cout << "Find the state." << endl;
        cout << "cost:  ";
        cout << tmp->cost_g << endl;
        //cout << tmp->state_str << endl;
        //cout << n << endl;
    }
}

int main()
{
    cout << "test input 1:  ";
    A_star_algorithm(state_input_str1);
    cout << "test input 2:  ";
    A_star_algorithm(state_input_str2);
    cout << "test input 3:  ";
    A_star_algorithm(state_input_str3);
    cout << "test input 4:  ";
    A_star_algorithm(state_input_str4);
    cout << "test input 5:  ";
    A_star_algorithm(state_input_str5);
    return 0;
}
