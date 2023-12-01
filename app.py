import streamlit as st
import pandas as pd
from SynRBL.SynExtract import RSMIProcessing, RSMIDecomposer, RSMIComparator
from SynRBL.SynRuleImpute import SyntheticRuleImputer
from SynRBL.rsmi_utils import load_database
from PIL import Image
from io import BytesIO
from SynRBL.SynVis import ReactionVisualizer

class RebalancingReaction:
    def __init__(self, rule_manager_pathname):
        self.rules = load_database(pathname=rule_manager_pathname)

    def process_reactions(self, input_data):
        if isinstance(input_data, str):
            input_data = [{'reactions': input_data}]
        elif not isinstance(input_data, list):
            raise ValueError("Input data must be a string or a list of dictionaries.")

        process = RSMIProcessing(data=pd.DataFrame(input_data), rsmi_col='reactions', parallel=True, n_jobs=-1, save_json=False)
        input_dict = process.data_splitter().to_dict('records')

        decompose = RSMIDecomposer(smiles=None, data=input_dict, reactant_col='reactants', product_col='products',
                                  parallel=True, n_jobs=-1, verbose=1)
        react_dict, product_dict = decompose.data_decomposer()

        comp = RSMIComparator(reactants=react_dict, products=product_dict, n_jobs=-1)
        unbalance, diff_formula = comp.run_parallel()

        reactions_clean = pd.concat([pd.DataFrame(input_dict),
                                     pd.DataFrame([unbalance]).T.rename(columns={0: 'Unbalance'}),
                                     pd.DataFrame([diff_formula]).T.rename(columns={0: 'Diff_formula'})],
                                    axis=1).to_dict(orient='records')

        imp = SyntheticRuleImputer(rule_dict=self.rules)
        dict_impute = imp.impute(missing_dict=reactions_clean[0:1])

        return dict_impute

# Create a Streamlit app
st.title("Reaction Rebalancing App")

# Sidebar
st.sidebar.header("Settings")
rule_manager_path = st.sidebar.text_input("Rule Manager Path:", "./Data/rule_manager.json.gz")
reaction_input = st.sidebar.text_area("Input Reaction:", "Nc1cccc2cnccc12.O=C(O)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1>>O=C(Nc1cccc2cnccc12)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1")
input_data_list = st.sidebar.text_area("Input Data (JSON format):", "")
input_data = []

if reaction_input:
    input_data.append({'reactions': reaction_input})

if input_data_list:
    input_data.extend(pd.read_json(input_data_list).to_dict('records'))

rebalancer = RebalancingReaction(rule_manager_path)
result_list = []

# Main content
if st.button("Process Reactions"):
    if not input_data:
        st.warning("Please provide reaction data.")
    else:
        result_list = rebalancer.process_reactions(input_data)
        st.subheader("Rebalanced Reaction:")
        st.write(result_list[0])

# Display the input data for reference
st.sidebar.header("Visualization")
if result_list:
    selected_result = st.sidebar.selectbox("Select Result:", result_list)
    st.sidebar.subheader("Visualize Reaction")
    visualizer = ReactionVisualizer(compare=True, orientation='vertical', figsize=(12, 6), label_position='above', dpi=300)
    
    # Change to use plot_reactions
    reaction_image = visualizer.plot_reactions(selected_result['reactions'], selected_result['new_reaction'])

    # Convert the PIL image to bytes
    image_bytes = BytesIO()
    reaction_image.save(image_bytes, format="PNG")

    # Clear the main content section and display the image
    st.empty()
    st.image(image_bytes, caption="Rebalanced Reaction", use_column_width=True)
