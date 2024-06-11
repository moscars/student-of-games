#pragma once

#include "publicBeliefState.h"

#include <memory>
#include <string>
#include <fstream>

namespace visualizeTree{

    template <class T>
    void graph_viz_helper(std::shared_ptr<T> node, std::ofstream& file){
        auto [name, label] = node->nodeToGraphViz();
        file << label << "\n";
        
        std::vector<std::string> children_names;
        for(auto child : node->getChildren()){
            auto [child_name, child_label] = child->nodeToGraphViz();
            children_names.push_back(child_name);
            graph_viz_helper(child, file);
        }


        for(auto child_name : children_names){
            file << name << " -> " << child_name << "\n";
        }
    }

    template <class T>
    void graph_viz_tree(std::shared_ptr<T> tree){
        std::ofstream file;

        // is statement based on the type
        if(std::is_same<T, PublicBeliefState>::value){
            file.open("tree.dot");
        } else{
            throw std::runtime_error("Type not supported");
        }

        file << "digraph G {\n";

        graph_viz_helper(tree, file);

        file << "}\n";
        file.close();
    }
};