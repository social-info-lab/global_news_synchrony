// compile first: g++ -std=c++11 -o c_extract_pair extract_pair.cpp
// command usage:
// sbatch --output=script_output/extract_pair/c_extract_pair_0_10.txt -e script_output/extract_pair/c_extract_pair_0_10.err c_extract_pair_script.sh 0 10


#include <vector>
#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <set>
#include <list>
#include <map>
#include <algorithm>
#include <math.h>
#include <ctime>
#include <unistd.h>
#include <cstring>

using namespace std;

int get_memory_by_pid(pid_t pid) {
  FILE* fd;
  char line[1024] = {0};
  char virtual_filename[32] = {0};
  char vmrss_name[32] = {0};
  int vmrss_num = 0;
  sprintf(virtual_filename, "/proc/%d/status", pid);
  fd = fopen(virtual_filename, "r");
  if(fd == NULL) {
    cout << "open " << virtual_filename << " failed" << endl;
    exit(1);
  }

  // VMRSS line is uncertain
  for(int i = 0 ; i < 60; i++) {
    if(strstr(line, "VMRSS:") != NULL) {
      sscanf(line, "%s %d", vmrss_name, &vmrss_num);
      break;
    }
  }
  fclose(fd);
  return vmrss_num;
}

int get_machine_memory() {
  const char* virtual_filename = "/proc/meminfo";
  FILE* fd;

  char line[1024] = {0};
  fd = fopen(virtual_filename, "r");
  if(fd == NULL) {
    cout << "open " << virtual_filename << " failed" << endl;
    exit(1);
  }
  char vmrss_name[32];
  int machine_memory;
  fgets(line, sizeof(line), fd);
  sscanf(line, "%s %d", vmrss_name, &machine_memory);
  fclose(fd);
  return machine_memory;
}

vector<string> split(const string& str, const string& delim) {
     vector<string> res;
     if("" == str) return res;
     //先将要切割的字符串从string类型转换为char*类型
     char * strs = new char[str.length() + 1] ; //不要忘了
     strcpy(strs, str.c_str());

     char * d = new char[delim.length() + 1];
     strcpy(d, delim.c_str());

     char *p = strtok(strs, d);
     while(p) {
          string s = p; //分割得到的字符串转换为string类型
          res.push_back(s); //存入结果数组
          p = strtok(NULL, d);
     }

     return res;
}


int main(int argc,char **argv) {
     // cout<<fixed;
     // cout.precision(16);

     map<string, int> lang_dict = {{"en", 0},{"de", 1},{"es", 2},{"pl", 3},{"zh", 4},{"fr", 5},{"ar", 6},{"tr", 7},{"it", 8},{"ru", 9}};
     //filter the top50 name entities shared by the most articles in the whole dataset, since they are just big nationalities, weekdays, Covid, and Trump
     int top_ne[50] = {408352, 148, 84263196, 28775762, 794, 25337, 668, 127, 18706971, 129, 130, 60, 105, 38, 128, 11746, 2570643, 132, 846570, 131, 30, 219311, 183, 29, 421211, 159, 228905, 2495862, 99, 89469904, 16, 3947, 25274, 46, 61, 463180, 96, 228563, 408, 142, 130879, 226494, 17, 956, 812, 23806, 7188, 43, 224475, 1033};
     map<int, int> top_ne_dict{};
     for (int i = 0; i < 50; ++i)
     {
        top_ne_dict[top_ne[i]] = 1;
     }
     int date_diff = 5;
     int start_date = stoi(argv[1]);
     int end_date = stoi(argv[2]);

     char buffer[256];
     char *val = getcwd(buffer, sizeof(buffer));
     if (val) {
         cout << buffer << endl;
     }

     cout << "starts from " << start_date << endl;
     cout << "ends at " << end_date << endl;


     int line_num = 0;
     unsigned long long int pair_num = 0;
     int ne_count_with_dup = 0;
     int cur_art_num = 1; //initialization to make pair_num = 0 since the beginning
     // set<unsigned long long int> pair_set;
     map<int, set<unsigned long long int>> pair_set_dict{};
     map <int, set<unsigned long long int>>::iterator iter;
     int cur_pair_set_dict_key;

     int cur_ne = 0; //the id of current name entity
     int cur_group_size = 0;
     int top_ne_flag = 0; // flag whether this line is for the top_ne's or not, if it is, skip this line
     string cur_lang_i;
     string cur_lang_j;
     int cur_lang_ind_i;
     int cur_lang_ind_j;
     int cur_digit_i;
     int cur_digit_j;
     int encode_dict_key;
     unsigned long long int encode_pair_key;
     time_t cur_time;
     unsigned long long int save_pair_num = 0;
     // int cur_max_line_num = 0;
     // int cur_max_digit = 0;

     ifstream infile;
     ofstream outfile;
     //infile.open("/Users/xichen/Documents/GitHub/mediacloud/ner_art_sampling/indexes/ne_art_indexes/top10-ne-art-wiki-filtered_0_10.index", ios::in);
     infile.open("indexes/ne_art_indexes/top10-ne-art-wiki-filtered_" + to_string(start_date) + "_" + to_string(end_date) + ".index", ios::in);

     if (!infile.is_open()) {
        cout << "failed to loading file..." << endl;
        return 1;
     }
     string buf;
     while (getline(infile, buf)) {
         line_num ++;
         if (top_ne_flag == 1){
             top_ne_flag = 0;
             continue;
         }

         // cout << buf << endl;
         if (line_num % 2 == 0){

             //loading the line
             std::replace(buf.begin(), buf.end(), '[', ' ');
             std::replace(buf.begin(), buf.end(), '(', ' ');
             std::replace(buf.begin(), buf.end(), ',', ' ');
             std::replace(buf.begin(), buf.end(), ')', ' ');
             std::replace(buf.begin(), buf.end(), ']', ' ');
             std::replace(buf.begin(), buf.end(), '\'', ' ');
             std::vector<string> cur_line = split(buf, " ");
             cur_group_size = int(cur_line.size()/3);

             int dat_list[cur_group_size];
             string ind_list[cur_group_size];
             float ind_line_list[cur_group_size];
             string ind_line_string_list[cur_group_size];

             for (int i = 0; i < cur_line.size(); ++i)
             {
               if (i%3 == 0){
                   dat_list[int(i/3)] = stoi(cur_line[i]);
               }
               else if (i%3 == 1){
                   ind_list[int(i/3)] = cur_line[i];
                   // cout<< i<< " "<<ind_list[int(i/3)] << " " << cur_line[i]<<endl;
               }
               else{
                   ind_line_list[int(i/3)] = (float)stoi(cur_line[i]);
                   ind_line_string_list[int(i/3)] = cur_line[i];
               }
             }

             // extract the pairs
             string cur_lang[cur_group_size];
             int cur_lang_ind[cur_group_size];
             int cur_digit[cur_group_size];


             for (int i = 0; i < cur_group_size; ++i){
                 cur_lang[i] = ind_list[i].substr(8,2);
                 cur_lang_ind[i] = lang_dict[cur_lang[i]];
                 cur_digit[i] = ceil(log10(ind_line_list[i] + 0.1)); //+0.1 to correctly computing the digit number of each original integer

                 // for debug
                 /*
                 if (ind_line_list[i]>100000){
                    if (cur_max_line_num < ind_line_list[i])
                        cur_max_line_num = ind_line_list[i];
                    if (cur_max_digit < cur_digit[i])
                        cur_max_digit = cur_digit[i];

                    cout << "digit: " << cur_digit[i] << " cur max digit: " << cur_digit[i] << "string :" << ind_line_string_list[i] << " line number: " << ind_line_list[i] << " current biggest: "<< cur_max_line_num << endl;
                 }
                 */
             }


             for (int i = 0; i < cur_group_size; ++i){
                 for (int j = 0; j < cur_group_size; ++j){
                    if ((i != j) && (abs(dat_list[i]-dat_list[j]) <= date_diff)){
                        if (ind_line_list[i] < ind_line_list[j]){
                            // encode_pair_key = (1000*cur_digit[i] + 100*cur_digit[j] + 10*cur_lang_ind[i] + cur_lang_ind[j]) * pow(10,cur_digit[i] + cur_digit[j]) + ind_line_list[i] * pow(10,cur_digit[j]) + ind_line_list[j];
                            encode_dict_key = 1000*cur_digit[i] + 100*cur_digit[j] + 10*cur_lang_ind[i] + cur_lang_ind[j];
                            encode_pair_key = ind_line_list[i] * pow(10,cur_digit[j]) + ind_line_list[j];
                        }
                        else if (ind_line_list[i] > ind_line_list[j]){
                            // encode_pair_key = (1000*cur_digit[j] + 100*cur_digit[i] + 10*cur_lang_ind[j] + cur_lang_ind[i]) * pow(10,cur_digit[j] + cur_digit[i]) + ind_line_list[j] * pow(10,cur_digit[i]) + ind_line_list[i];
                            encode_dict_key = 1000*cur_digit[j] + 100*cur_digit[i] + 10*cur_lang_ind[j] + cur_lang_ind[i];
                            encode_pair_key = ind_line_list[j] * pow(10,cur_digit[i]) + ind_line_list[i];
                        }
                        else{
                            if (cur_lang_ind[i] <= cur_lang_ind[j]){
                                // encode_pair_key = (1000*cur_digit[i] + 100*cur_digit[j] + 10*cur_lang_ind[i] + cur_lang_ind[j]) * pow(10,cur_digit[i] + cur_digit[j]) + ind_line_list[i] * pow(10,cur_digit[j]) + ind_line_list[j];
                                encode_dict_key = 1000*cur_digit[i] + 100*cur_digit[j] + 10*cur_lang_ind[i] + cur_lang_ind[j];
                                encode_pair_key = ind_line_list[i] * pow(10,cur_digit[j]) + ind_line_list[j];
                            }
                            else{
                                // encode_pair_key = (1000*cur_digit[j] + 100*cur_digit[i] + 10*cur_lang_ind[j] + cur_lang_ind[i]) * pow(10,cur_digit[j] + cur_digit[i]) + ind_line_list[j] * pow(10,cur_digit[i]) + ind_line_list[i];
                                encode_dict_key = 1000*cur_digit[j] + 100*cur_digit[i] + 10*cur_lang_ind[j] + cur_lang_ind[i];
                                encode_pair_key = ind_line_list[j] * pow(10,cur_digit[i]) + ind_line_list[i];
                            }
                        }

                        // pair_set.insert(encode_pair_key);
                        pair_set_dict[encode_dict_key].insert(encode_pair_key);

                        pair_num++;


                        if (pair_num%1000000==0){

                            cur_time = time(0);
                            char* dt = ctime(&cur_time);
                            cout << "processed " << line_num << " lines, processed " << pair_num << " pairs, current time is:" << dt << endl;

                            // cout << "The current process consumes " << int(sizeof(pair_set)/1024/1024) << "MB memory" << endl;
                            // cout << "The machine memory: " << get_machine_memory() / 1024 << "MB memory" << endl;
                        }

                    }
                 }
             }

          }
          else{
             std::vector<string> cur_line = split(buf, " ");

             pair_num += cur_art_num * (cur_art_num - 1);
             cur_ne = stoi(cur_line[0]);
             cur_art_num = stoi(cur_line[1]);
             ne_count_with_dup += cur_art_num;

             // flag whether this line is for the top_ne's or not, if it is, skip this line
             if (top_ne_dict.find(cur_ne) != top_ne_dict.end()){
                 top_ne_flag = 1;
             }
          }
      }

     infile.close();

     // saving the pairs to files.
     outfile.open("network_pairs/pairs-top10-ne-art-wiki-filtered_" + to_string(start_date) + "_" + to_string(end_date) + ".txt", ios::out);
     /*
      for ( auto it = pair_set.begin(); it != pair_set.end(); it++ ){
         save_pair_num++;
         outfile << *it << endl;
         if (save_pair_num%1000==0){
            cout << "saved " << save_pair_num << " pairs..." << endl;
         }

     }
     */

     for(iter=pair_set_dict.begin(); iter!=pair_set_dict.end(); iter++)
     {
          for ( auto it = iter->second.begin(); it != iter->second.end(); it++ ){
              save_pair_num++;
              outfile << iter->first << " "<< *it << endl;
              if (save_pair_num%10000==0){
                 cout << "saved " << save_pair_num << " pairs..." << endl;
              }
          }
     }

     outfile.close();
     cout << "finished extracting the pairs....." << endl;

     return 0;
}
