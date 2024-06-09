import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
// FINAL VERSION
// Code of the epsBE.jar to compute the embedding of network
public class Valmari {
    
    
    public static LinkedList<Block> fillUb(List<State> ls){
        
        LinkedList<Block> result = new LinkedList<Block>();
        
        
        for(int i = 0; i<ls.size();i++) {
            
            State s = ls.get(i);
            Block ofs = s.block;
            if(ofs.InUb==false) {
                result.add(ofs);
                ofs.setInUb(true);
            }
        }
        
        return result;
    }
    
    public static List<List<Integer>> buildPartition(List<State> ls) {
        
        List<List<Integer>> result = new LinkedList<List<Integer>>();
        for (int k = 0; k<ls.size();k++) {
            List<Integer> temp = new LinkedList<Integer>();
	        for(int i = 0; i<ls.size();i++) {
	            if(k==ls.get(i).block.number) {

	                temp.add(ls.get(i).number );
	            }
	        }
            if(temp.size()>0) {
                result.add(temp);
            }
        }
        return result;
        
    }
    
    

    

    public static int reduceBE(LinkedList<Block> Ub, List<State> S, List<List<Integer>> toS, int count, double eps, int[] initialPartitionArray) {
        

        while(Ub.size()!=0) {
            
            Block Bi = Ub.pop();
 
            Bi.setInUb(false);
            ArrayList<State> St = new ArrayList<State>();
            LinkedList<Block> Bt = new LinkedList<Block>();
            
            
            LinkedList<State> sprimo = Bi.getStates();
            
            Iterator it = sprimo.iterator();
              
            while(it.hasNext()){
                
                State s = (State)it.next();
                List<Integer> ss = toS.get(s.getNumber()-1);
                
                for(int i = 0; i<ss.size(); i++) {
                    
                    int ind = ss.get(i)-1;
                    
                    if(S.get(ind).w==0) {
                        St.add(S.get(ind));
                        S.get(ind).setW(1);
                    }
                    else {
                        S.get(ind).setW(S.get(ind).w+1);
                    }
                    
                }
                

                
                
                
            }
            

            
            for(int i = 0; i<St.size();i++) {
                
                State s = St.get(i);
                Block Bs = s.getBlock();
                if(Bs.marked==false && Bs.getStates().size()>1) {
                    
                    Bt.add(Bs);
                    
                }
                Bs.setMarked(true);
                
            }
            

            
            while(Bt.size()!=0) {
                
                Block B = Bt.pop();
                boolean inn = B.InUb;
                
                B.PMCsorting();
                
                
    
                
                List<List<State>> blocks = B.partition(B.states, eps);
                
            

                
                LinkedList<Block> provaBlock = new LinkedList<Block>();
                
                boolean first = true;
                if(blocks.size()>1) {
                for(int i = 0;i<blocks.size();i++) {
                    
                    if(first==false) {
                        
                        Block Bl = new Block(count);
                        count = count+1;
      
        				Bl.setStates((LinkedList)blocks.get(i));
        	
        				provaBlock.add(Bl);

                    }
                    else {
                        
        				
        				B.setStates((LinkedList)blocks.get(i));
                        if(B.InUb==false) {
                            provaBlock.add(B);

                        }
                        first = false;
                        
                    }
                    
                }
                }
                
                if(provaBlock.size()>0) {
                for(int i = 0; i<provaBlock.size();i++) {
                    Block Bl = provaBlock.get(i);
                    Ub.add(Bl);
                    Bl.setInUb(true);
                }
                }
                B.setMarked(false);
            }
            

            
            Iterator itS = S.iterator();
            while(itS.hasNext()){
                
                State sg = (State) itS.next();
                sg.setW(0);
            
            }


        }
        

        return count;
        
    }
    
    public static void  writeFile(FileWriter myWriter, int[] array) throws IOException {
        
        
        boolean found = false;
        int m = max(array);
        String totalS = "";
        for(int i =0; i<=m;i++) {
            found=false;
            String s = "{ ";
            for(int j = 0; j<array.length; j++) {
                
                if(i == array[j]) {
                    
                    s = s+"x"+(j+1)+ " ";
                    found = true;
                }
                
            }
            s= s+"},";
            totalS = totalS+s;
            if(found==true) {

                myWriter.write(s);
            }
        }

        myWriter.write("\n\n");



    }
    
    public static int max(int[] array) {
        
        int m = 0;
        for(int i=0; i < array.length;i++) {
            
            if(array[i]>m) {
                
                m = array[i];
                
            }
            
        }
        return m;
        
    }

    public static int[][] readMatrix(String name, int N) {
    	
    	int[][] mat = new int[N][N];
        try {
            File myObj = new File(name);
            Scanner myReader = new Scanner(myObj);
            while (myReader.hasNextLine()) {
              String data = myReader.nextLine();
              String[] sdata = data.split(" ");
              //System.out.println(sdata[0]+" "+sdata[1]);
              mat[Integer.parseInt(sdata[0])][Integer.parseInt(sdata[1])] = 1;
              mat[Integer.parseInt(sdata[1])][Integer.parseInt(sdata[0])] = 1;
            }
            myReader.close();
          } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
          }
    	return mat;
    	
    }
    
    
    
    public static ArrayList<List<Integer>> fillST(String name,int N) {
    	
    	ArrayList<List<Integer>> fillS = new ArrayList<List<Integer>>(N);
    	for(int i = 0; i<N;i++) {
    		
    		fillS.add(new LinkedList<Integer>());
    		
    	}
        try {
            File myObj = new File(name);
            Scanner myReader = new Scanner(myObj);
            while (myReader.hasNextLine()) {
              String data = myReader.nextLine();
              String[] sdata = data.split(" ");

              int i = Integer.parseInt(sdata[0]);
              int j = Integer.parseInt(sdata[1]);
              if(i!=j) {
            	  fillS.get(i).add(j+1);
            	  fillS.get(j).add(i+1);
              }
              else {
            	  fillS.get(i).add(j+1);
              }
            }
            myReader.close();
          } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
          }
    	return fillS;
    }
    
    public static void main(String[] args) throws IOException {
        
        int N = Integer.parseInt(args[1]);
        String nname = args[0];
        int count = 1;
        int counterIt = 0;
        
        
        double eps = Double.parseDouble(args[2]);
        double EPS = Double.parseDouble(args[3]);
        double step = Double.parseDouble(args[4]);
        
        boolean prePart=false;
        String pname = "null";
        if(!pname.equals("null")) {
        	prePart=true;
        }


        List<List<Integer>> toS = fillST("datasets/"+nname+".edgelist",N);
        
        for(int i=0; i< N; i++) {
        	Collections.sort(toS.get(i));
        }

        List<List<Integer>> toST = toS;
    	int [] initialPartitionArray = new int [N];    	
    	if(prePart==true) {
    		Scanner scanner = new Scanner(new File(pname));
    	int cc = 0;
    	while(scanner.hasNextInt()){
    	   initialPartitionArray[cc++] = scanner.nextInt();
    	}
    	}
    	else {
    		Arrays.fill(initialPartitionArray, 1);
    	}

    	int mmax = max(initialPartitionArray);
    	ArrayList<LinkedList<State>> inStore = new ArrayList<LinkedList<State>>(mmax);
    	
    	List<State> S = new ArrayList<State>();
    	for(int i=0;i<N;i++) {
    		
    		State st = new State(i+1,null);
    		st.setW(0);
    		S.add(st);
    		
    	}
    	
    	for(int i = 0; i<mmax;i++) {
    		 
    		inStore.add(i,new LinkedList<State> ());
    		
    	}
    	
    	
    	for(int i = 0; i<N;i++) {
    		
    		inStore.get(initialPartitionArray[i]-1).add(S.get(i));
    		
    	}
    	
    	LinkedList<Block> Ub = new LinkedList<Block>();

    	
    	for(int i = 0; i<mmax; i++) {
    		
    		Block Bi = new Block(count);
    		count=count+1;
    		Bi.setStatesIIn(inStore.get(i));
    		Ub.add(Bi);
    		Bi.setInUb(true);
    		
    	}
    	
        long start = System.currentTimeMillis();
        
        while(eps <= EPS) {
            
        int lenA = 0;
        int lenAT = count;
        counterIt = 0;
        
        
        Valmari.reduceBE(Ub,S,toS,lenAT,eps,initialPartitionArray);
        List<List<Integer>> lt = Valmari.buildPartition(S);

        count = 0;
        if(eps+step<=EPS) {
        	
        	inStore = new ArrayList<LinkedList<State>>(mmax);
        	for(int i = 0; i<mmax;i++) {
        		
        		inStore.add(i,new LinkedList<State>());
        		
        	}
            for(int i = 0;i<N;i++) {
                
                State si = S.get(i);
                Block bb = si.block;
                if(bb.states.size()==1) {
                    
                	inStore.get(si.inBlock-1).add(si);

                    
                }
                else if(bb.InUb==false){
                    
                    bb.number=count;
                    count = count+1;
                    Ub.add(bb);
                    bb.InUb=true;
                    
                }
                
            }
            
            for(int z=0;z<inStore.size();z++) {
            	
            	if(inStore.get(z).size()>0) {
            		
            		Block Bi = new Block(count);
            		count=count+1;
            		Bi.setStates(inStore.get(z));
            		Ub.add(Bi);
            		Bi.InUb=true;
            		
            	}
            	
            }

        }
        
        eps = eps + step;
        }

        long end = System.currentTimeMillis();
        List<List<Integer>> lt = Valmari.buildPartition(S);
        
    
        int[] part = new int[N];
        for(int i = 0;i<N;i++) {
            part[i] = S.get(i).block.number;
        }
        
        FileWriter myWriter = null;
        myWriter = new FileWriter("./embedNEW/"+nname+"BE");
        myWriter.write("PARTIZIONE OBTAINED FINALE \n");
        writeFile(myWriter,part);   
        myWriter.close();
        System.out.println("Embedding size " + lt.size());

    }
    
    
    
    
}