install.packages("igraph")
install.packages("gplots")
install.packages("rjson")
library(rjson)
library(igraph)
library("gplots")

setwd("C:/Users/Sebi/PycharmProjects/network-analysis-of-defi-protocols")

categories = fromJSON(file = "./data/input/protocols-per-category.json")

centralities <- c("pagerank", "katz", "eigenvector", "closeness", "betweenness")

# make sure that the output paths exist before plotting!
#"./results/plots/heatmaps/svg/"
#"./results/plots/heatmaps/png/"

# choose png or svg output file
format = "svg"

size = 16
ratio = 0.5625

for (protocols in categories) {
  for (protocol in protocols) {
    file = paste("./data/processed/defi-networks/",protocol,".graphml",sep="")
    plot = "./results/plots/heatmaps/"
    
    g <- read_graph(
      file,
      format = "graphml"
    )
    
    g_df = as_data_frame(g, what = c("vertices"))
    
    g_df_cent <- g_df[c("pagerank", "katz", "eigenvector", "closeness", "betweenness", "id", "etherscan", "label")]
    
    g_df_cent[g_df_cent==""]<-NA
    
    df_ids <- g_df[c("id")]
    
    top_nodes <- data.frame()
    
    # TODO: fix ordering of the centrality values (Top 10 values for each ...)
    for (centrality in centralities) {
      ordered_cent = g_df_cent[order(g_df_cent[centrality], decreasing = TRUE),]
      top_nodes <- rbind(top_nodes, head(ordered_cent, 10))
    }
    
    top_nodes = unique(top_nodes)
    
    row.names(g_df_cent) <- g_df_cent$id
    
    row.names(top_nodes) <- paste(ifelse((top_nodes$etherscan == 'no label' | is.na(top_nodes$etherscan)), 
                                  ifelse(is.na(top_nodes$label), 'no label', top_nodes$label), top_nodes$etherscan), ': ',
                                  top_nodes$id, sep="") 
    
    top_nodes$id <- NULL
    top_nodes$etherscan <- NULL
    top_nodes$label <- NULL
    
    data<-as.matrix(apply(top_nodes,2 , rev))
    
    if (format=="svg"){
      svg(paste(plot, "/svg/",protocol,".svg",sep=""), width = size, height = size * ratio)
    }
    
    if (format=="png"){
      png(paste(plot, "/png/",protocol,".png",sep=""), width = size, height = size * ratio, units="in", res=162)
    }
    
    rows = dim(data)[1]
  
    if(rows > 50) rows = 50
    
    color <- rev(heat.colors(100))
    
    heatmap.2(data[1:rows,], Colv = NA, Rowv = NA, dendrogram='none', main=protocol,
              margins=c(10,size*2.4), srtCol=45, trace="none", col=color, key=TRUE)
    
    dev.off()
    
    print(protocol)
  }
}

